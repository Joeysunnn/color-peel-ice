"""Bounded Scheme B: ConceptExpress coarse concepts followed by three-point SAM."""

import argparse
import gc
from importlib.metadata import version
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods.colorpeel_ice.object_segmentation import (
    SAM_MODEL, SAM_REVISION, save_object, sha256, write_json,
)


def sample_points(mask, rng):
    """Three distinct uniformly sampled foreground pixels, in native x/y order."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        raise ValueError('coarse mask has fewer than three foreground pixels')
    indices = rng.choice(len(xs), size=3, replace=False)
    return np.stack([xs[indices], ys[indices]], axis=1).tolist()


class PointSam:
    def __init__(self, device):
        import torch
        from transformers import SamModel, SamProcessor

        self.torch, self.device = torch, device
        self.processor = SamProcessor.from_pretrained(
            SAM_MODEL, revision=SAM_REVISION, local_files_only=True)
        self.model = SamModel.from_pretrained(
            SAM_MODEL, revision=SAM_REVISION, local_files_only=True,
            torch_dtype=torch.float32).to(device).eval()
        self.receipt = {'model': SAM_MODEL, 'requested_revision': SAM_REVISION,
                        'loaded_revision': getattr(self.model.config, '_commit_hash', None),
                        'dtype': 'float32', 'local_files_only': True,
                        'selection': 'maximum predicted IoU; no GT or coarse overlap ranking'}

    def __call__(self, image, points):
        inputs = self.processor(images=image, input_points=[[points]],
                                input_labels=[[[1, 1, 1]]], return_tensors='pt').to(self.device)
        with self.torch.no_grad():
            result = self.model(**inputs, multimask_output=True)
        masks = self.processor.image_processor.post_process_masks(
            result.pred_masks.cpu(), inputs['original_sizes'].cpu(),
            inputs['reshaped_input_sizes'].cpu())[0][0]
        scores = result.iou_scores[0, 0].cpu()
        best = int(scores.argmax())
        return masks[best].bool().numpy(), scores.tolist(), best


def refine(image, masks64, output, seed, segmenter):
    from src.methods.colorpeel_ice.conceptexpress_masks import mask_to_native

    rng = np.random.default_rng(seed)
    rows, failures = [], []
    for index, mask64 in enumerate(masks64):
        concept_id = f'concept_{index + 1:03d}'
        native = mask_to_native(mask64, image.size)
        try:
            points = sample_points(native, rng)
            write_json(output / 'coarse' / f'{concept_id}_points.json', points)
            preview = image.copy()
            draw = ImageDraw.Draw(preview)
            for x, y in points:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill='red', outline='white')
            preview.save(output / 'coarse' / f'{concept_id}_points.png')
            mask, scores, selected = segmenter(image, points)
            if mask.shape != native.shape or not mask.any():
                raise ValueError('empty or incorrectly sized SAM mask')
            intersection = int(np.logical_and(native, mask).sum())
            row = save_object(image, {'mask': mask, 'category': concept_id,
                'positive_points_xy': points, 'sam_candidate_scores': scores,
                'selected_candidate': selected, 'sam_iou_score': scores[selected],
                'coarse_refined_iou': intersection / int(np.logical_or(native, mask).sum())},
                output, concept_id)
            rows.append(row)
        except Exception as error:
            failures.append({'concept_id': concept_id, 'error': f'{type(error).__name__}: {error}'})
    return rows, failures


def run(args):
    from scripts.launch.colorpeel_run import git_info, resolve_run_dir
    from experiments.sd21_qwen_initializers_v1.run import require_idle
    from experiments.sd21_wood_branches_v1.prepare_model import verify_model
    from src.methods.colorpeel_ice.conceptexpress_masks import extract, mask_to_native

    image_path = args.image.resolve()
    output = resolve_run_dir(args.output)
    if output.exists() or any(output.is_relative_to(path) or path.is_relative_to(output)
                              for path in (ROOT, args.model_dir.resolve(),
                                           args.upstream_checkout.resolve(), image_path)):
        raise ValueError('require a fresh external output separate from inputs and code')
    revision = git_info()
    output.mkdir(parents=True)
    state = {'scheme': 'B', 'status': 'started', 'git': revision, 'seed': args.seed,
             'command': sys.argv, 'source': str(image_path), 'source_sha256': sha256(image_path),
             'gpu': args.gpu, 'training_started': False, 'gt_used_for_prediction': False,
             'quality_accepted_by_user': False, 'objects': [], 'failures': []}
    write_json(output / 'status.json', state)
    try:
        require_idle((args.gpu,))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', DIFFUSERS_OFFLINE='1')
        state['base_model'] = verify_model(args.model_dir)['model']
        state['packages'] = {name: version(name) for name in
                             ('torch', 'diffusers', 'transformers', 'numpy', 'scipy', 'scikit-learn')}
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert('RGB')
            image.load()
        image.save(output / 'source.png')
        state['canonical_source_sha256'] = sha256(output / 'source.png')
        masks64, diagnostics = extract(image, args.model_dir, args.upstream_checkout,
                                       device='cuda:0', seed=args.seed)
        arrays = {key: value for key, value in diagnostics.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(output / 'attention_diagnostics.npz', **arrays)
        state['extraction'] = {key: value for key, value in diagnostics.items() if key not in arrays}
        state['coarse_count'] = len(masks64)
        coarse = output / 'coarse'
        coarse.mkdir()
        image.save(coarse / 'source.png')
        for index, mask64 in enumerate(masks64):
            concept_id = f'concept_{index + 1:03d}'
            Image.fromarray(np.uint8(mask64) * 255).save(coarse / f'{concept_id}_64.png')
            save_object(image, {'mask': mask_to_native(mask64, image.size),
                               'category': concept_id}, coarse, concept_id)
        state['status'] = 'coarse_complete'
        write_json(output / 'status.json', state)
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        sam = PointSam('cuda:0')
        state['sam'] = sam.receipt
        state['objects'], state['failures'] = refine(image, masks64, output, args.seed, sam)
        if sha256(image_path) != state['source_sha256'] or git_info() != revision:
            raise ValueError('source image or checkout changed during extraction')
        state['status'] = ('partial' if state['failures'] else 'succeeded') if state['objects'] else 'failed'
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(output / 'status.json', state)
        write_json(output / 'objects.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upstream-checkout', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=Path(
        '/var/tmp/colorpeel-r12user5/assets/sd21_base_4e63672_common_checker'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20)
    state = run(parser.parse_args())
    print(state['status'], 'coarse:', state.get('coarse_count'), 'refined:', len(state['objects']))
    return int(state['status'] != 'succeeded')


if __name__ == '__main__':
    sys.exit(main())
