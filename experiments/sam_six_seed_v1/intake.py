"""Prompt cached SAM with six fixed GT bounding boxes, retaining unmodified SAM masks."""

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from experiments.icbench_gt_mesh_v1.run import validate_selection
from scripts.launch.colorpeel_run import git_info
from src.methods.colorpeel_ice.object_segmentation import SAM_MODEL, SAM_REVISION, sha256, write_json
from src.methods.colorpeel_ice.single_image_pipeline import read
from src.methods.colorpeel_ice.single_image_subject import check_locks, fresh, prepare_inputs

OBJECT_KEYS = ('image_000__mask_160', 'image_001__mask_008', 'image_003__mask_207',
               'image_009__mask_205', 'image_046__mask_097', 'image_081__mask_130')


class BoxSAM:
    def __init__(self):
        import torch
        from transformers import SamModel, SamProcessor
        from transformers.utils.hub import cached_file

        self.torch = torch
        self.processor = SamProcessor.from_pretrained(SAM_MODEL, revision=SAM_REVISION, local_files_only=True)
        self.model = SamModel.from_pretrained(SAM_MODEL, revision=SAM_REVISION,
            torch_dtype=torch.float32, local_files_only=True).to('cuda:0').eval()
        resolved = getattr(self.model.config, '_commit_hash', None)
        if resolved not in (None, SAM_REVISION):
            raise ValueError('loaded SAM revision differs from the pinned revision')
        configs = [Path(cached_file(SAM_MODEL, name, revision=SAM_REVISION, local_files_only=True))
                   for name in ('config.json', 'preprocessor_config.json')]
        self.receipt = {'model': SAM_MODEL, 'requested_revision': SAM_REVISION, 'resolved_revision': resolved,
            'device': 'cuda:0', 'dtype': 'float32', 'local_files_only': True, 'multimask_output': True,
            'packages': {name: version(name) for name in ('torch', 'transformers')},
            'source_locks': {str(path): sha256(path) for path in configs}}

    def __call__(self, image, box):
        inputs = self.processor(images=image, input_boxes=[[box]], return_tensors='pt').to('cuda:0')
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype=self.torch.float32)
        with self.torch.no_grad():
            outputs = self.model(**inputs, multimask_output=True)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.detach().cpu(),
            inputs['original_sizes'].detach().cpu(), inputs['reshaped_input_sizes'].detach().cpu())[0][0]
        return masks.to(self.torch.bool).numpy(), outputs.iou_scores.detach().cpu()[0, 0].numpy()


def prompt_box(image, gt):
    """GT pixels are used only to compute an exclusive bounding box."""
    if gt.shape != (image.height, image.width) or not set(np.unique(gt)) <= {0, 255}:
        raise ValueError('GT prompt source must be an aligned binary mask')
    yy, xx = np.nonzero(gt)
    if not len(xx):
        raise ValueError('empty GT prompt source')
    return [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]


def run(selection_path, output, segmenter=None):
    selection_path = Path(selection_path).resolve()
    selection = read(selection_path)
    locks = validate_selection(selection)
    by_key = {row['object_key']: row for row in selection['objects']}
    if not set(OBJECT_KEYS) <= set(by_key):
        raise ValueError('selection lacks one of the six fixed GT identities')
    revision, output = git_info(), fresh(output)
    locks[str(selection_path)] = sha256(selection_path)
    state = {'schema': 'sam_six_seed_inputs/v1', 'git': revision, 'status': 'running',
        'protocol': 'GT-box-prompted SAM; GT pixels only define exclusive prompt bbox; predicted SAM IoU argmax',
        'gt_mask_clipping': False, 'gt_iou_selection': False, 'mask_repair': False,
        'training_started': False, 'quality_accepted_by_user': False,
        'source_locks': locks, 'objects': [{'object_key': key, 'status': 'pending'} for key in OBJECT_KEYS]}
    manifest_path = output / 'inputs.json'
    write_json(manifest_path, state)
    try:
        segmenter = BoxSAM() if segmenter is None else segmenter
        model = getattr(segmenter, 'receipt', {'backend': 'injected'})
        write_json(output / 'sam_model.json', model)
        locks.update(model.get('source_locks', {}))
        locks[str(output / 'sam_model.json')] = sha256(output / 'sam_model.json')
        for item in state['objects']:
            key = item['object_key']
            source = by_key[key]
            folder = output / key
            folder.mkdir()
            report = {'object_key': key, 'status': 'prompting', 'image': source['image'],
                'gt_prompt_source': source['mask'], 'model_receipt': str(output / 'sam_model.json'),
                'selection': 'argmax SAM predicted IoU; first index on ties',
                'gt_pixels_used_for': 'exclusive bounding box only', 'mask_repair': False}
            try:
                check_locks({source[k]['path']: source[k]['sha256'] for k in ('image', 'mask')})
                with Image.open(source['image']['path']) as opened, Image.open(source['mask']['path']) as gt:
                    image = ImageOps.exif_transpose(opened).convert('RGB')
                    box = prompt_box(image, np.asarray(gt.convert('L')))
                report['prompt_bbox_xyxy_exclusive'] = box
                masks, scores = (np.asarray(value) for value in segmenter(image, box))
                raw_path = folder / 'sam_outputs.npz'
                np.savez_compressed(raw_path, masks=masks, predicted_iou_scores=scores)
                locks[str(raw_path)] = sha256(raw_path)
                if (masks.shape != (3, image.height, image.width) or masks.dtype.kind not in 'buif'
                        or not np.isfinite(masks).all() or not set(np.unique(masks)) <= {0, 1}
                        or scores.shape != (3,) or not np.isfinite(scores).all()):
                    raise ValueError('SAM must return three aligned binary candidates and finite predicted IoU scores')
                selected = int(scores.argmax())
                report.update(predicted_iou_scores=scores.tolist(), selected_index=selected, candidates=[])
                for index, mask in enumerate(masks):
                    path = folder / f'sam_candidate_{index}.png'
                    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
                    locks[str(path)] = sha256(path)
                    report['candidates'].append({'path': str(path), 'sha256': locks[str(path)],
                                                 'mask_pixels': int(mask.sum())})
                selected_path = folder / 'sam_selected.png'
                Image.fromarray(masks[selected].astype(np.uint8) * 255).save(selected_path)
                locks[str(selected_path)] = sha256(selected_path)
                if not masks[selected].any():
                    raise ValueError('selected SAM mask is empty; no alternate candidate fallback')
                receipt = prepare_inputs(source['image']['path'], selected_path, folder / 'input',
                    segmentation=f'GT-box-prompted SAM for {key}; GT pixels only supply bbox; unmodified SAM mask')
                report['status'] = 'prepared'
                receipt_path = folder / 'input/source_receipt.json'
                item.update(status='prepared', source_receipt=str(receipt_path), source_receipt_sha256=sha256(receipt_path))
                locks.update(receipt['source_locks'])
                locks[str(receipt_path)] = item['source_receipt_sha256']
            except Exception as error:
                report.update(status='blocked', error=f'{type(error).__name__}: {error}')
                item.update(status='blocked', blocked_reason=report['error'])
            write_json(folder / 'segmentation.json', report)
            locks[str(folder / 'segmentation.json')] = sha256(folder / 'segmentation.json')
            write_json(manifest_path, state)
        check_locks(locks)
        if git_info() != revision:
            raise ValueError('checkout changed during SAM intake')
        state['status'] = 'prepared' if all(item['status'] == 'prepared' for item in state['objects']) else 'partial'
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        for item in state['objects']:
            if item['status'] == 'pending':
                item.update(status='blocked', blocked_reason=state['error'])
        raise
    finally:
        write_json(manifest_path, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.selection, args.output)
