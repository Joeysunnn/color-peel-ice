"""Independent object masks for an EXIF-corrected natural photograph."""

import hashlib
import html
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import numpy as np
from PIL import Image, ImageOps

from scripts.methods.colorpeel_ice.segment_grounded_sam import (
    BOX_THRESHOLD, GROUNDING_DINO_MODEL, SAM_MODEL, TEXT_THRESHOLD,
)


SCHEMA = 'colorpeel_object_segmentation/v1'
DEDUPLICATION_IOU = 0.9
DEDUPLICATION_OVERLAP_MIN = 0.9
GROUNDING_REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'
SAM_REVISION = '70c1a07f894ebb5b307fd9eaaee97b9dfc16068f'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf8')


class ObjectSegmenter:
    """Load pinned cached models, retaining each detection as its own instance."""

    def __init__(self, device='cuda:0'):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

        self.torch, self.device, self.torch_dtype = torch, device, torch.float32
        self.grounding_processor = AutoProcessor.from_pretrained(
            GROUNDING_DINO_MODEL, revision=GROUNDING_REVISION, local_files_only=True)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_MODEL, revision=GROUNDING_REVISION,
            torch_dtype=torch.float32, local_files_only=True).to(device).eval()
        self.sam_processor = SamProcessor.from_pretrained(
            SAM_MODEL, revision=SAM_REVISION, local_files_only=True)
        self.sam_model = SamModel.from_pretrained(
            SAM_MODEL, revision=SAM_REVISION,
            torch_dtype=torch.float32, local_files_only=True).to(device).eval()
        packages = {}
        for name in ('torch', 'transformers', 'Pillow', 'numpy'):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        self.receipt = {
            'grounding_model': GROUNDING_DINO_MODEL,
            'grounding_requested_revision': GROUNDING_REVISION,
            'grounding_revision': getattr(self.grounding_model.config, '_commit_hash', None),
            'sam_model': SAM_MODEL,
            'sam_requested_revision': SAM_REVISION,
            'sam_revision': getattr(self.sam_model.config, '_commit_hash', None),
            'device': device, 'dtype': 'float32', 'local_files_only': True,
            'box_threshold': BOX_THRESHOLD, 'text_threshold': TEXT_THRESHOLD,
            'python': platform.python_version(), 'packages': packages,
        }

    def __call__(self, image, category):
        torch = self.torch
        inputs = self.grounding_processor(
            images=image, text=category + '.', return_tensors='pt').to(self.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(dtype=self.torch_dtype)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)
        detection = self.grounding_processor.post_process_grounded_object_detection(
            outputs, input_ids=inputs.input_ids, box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD, target_sizes=[image.size[::-1]])[0]
        boxes = detection['boxes'].detach().cpu()
        if not len(boxes):
            return []
        sam_inputs = self.sam_processor(
            images=image, input_boxes=[boxes.tolist()], return_tensors='pt').to(self.device)
        sam_inputs['pixel_values'] = sam_inputs['pixel_values'].to(dtype=self.torch_dtype)
        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs, multimask_output=True)
        masks = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.detach().cpu(), sam_inputs['original_sizes'].detach().cpu(),
            sam_inputs['reshaped_input_sizes'].detach().cpu())[0]
        scores = sam_outputs.iou_scores.detach().cpu()[0]
        best = scores.argmax(dim=-1)
        return [{
            'category': category, 'box': box.tolist(),
            'detection_score': float(detection['scores'][index].detach().cpu()),
            'sam_iou_score': float(scores[index, best[index]]),
            'mask': masks[index, best[index]].to(torch.bool).numpy(),
        } for index, box in enumerate(boxes)]


def validate_candidate(candidate, category, size, candidate_id):
    width, height = size
    if not isinstance(candidate, dict) or candidate.get('category', category) != category:
        raise ValueError('candidate category differs from its detection query')
    mask = np.asarray(candidate['mask'])
    if mask.shape != (height, width):
        raise ValueError('mask_size_mismatch')
    if mask.dtype.kind not in 'buif' or not np.isfinite(mask).all():
        raise ValueError('mask must be finite binary pixels')
    values = set(np.unique(mask).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError('mask must be binary')
    mask = mask != 0
    if not mask.any():
        raise ValueError('empty_mask')
    box = np.asarray(candidate['box'], dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError('box must contain four finite coordinates')
    box = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError('empty_detection_box')
    scores = {key: float(candidate[key]) for key in ('detection_score', 'sam_iou_score')}
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError('nonfinite_detection_or_sam_score')
    if not 0 <= scores['detection_score'] <= 1:
        raise ValueError('detection_score must be between zero and one')
    return {'candidate_id': candidate_id, 'category': category, 'box': box.tolist(),
            **scores, 'mask_pixels': int(mask.sum()), 'mask_ratio': float(mask.mean()), 'mask': mask}


def save_object(image, candidate, output, object_id):
    folder = output / 'objects' / object_id
    folder.mkdir(parents=True)
    mask = candidate['mask']
    rgb = np.asarray(image)
    ys, xs = np.nonzero(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    padding = max(1, int(np.ceil(0.05 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]))))
    crop = [max(0, bbox[0] - padding), max(0, bbox[1] - padding),
            min(image.width, bbox[2] + padding), min(image.height, bbox[3] + padding)]
    alpha = mask.astype(np.uint8) * 255
    cutout = np.concatenate([np.where(mask[..., None], rgb, 0), alpha[..., None]], axis=-1)
    overlay = rgb.copy()
    overlay[mask] = np.rint(0.65 * rgb[mask] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
    masked_rgb = np.where(mask[..., None], rgb, 127).astype(np.uint8)
    images = {
        'object_mask': Image.fromarray(alpha),
        'cutout': Image.fromarray(cutout),
        'overlay': Image.fromarray(overlay),
        'vlm_input': Image.fromarray(masked_rgb).crop(crop),
    }
    row = {key: value for key, value in candidate.items() if key != 'mask'}
    row.update(id=object_id, mask_pixels=int(mask.sum()), mask_ratio=float(mask.mean()),
               mask_bbox_xyxy=bbox, crop_box_xyxy=crop, source_image=str(output / 'source.png'))
    for name, artifact in images.items():
        path = folder / (name + '.png')
        artifact.save(path)
        row[name] = str(path)
        row[name + '_sha256'] = sha256(path)
    return row


def run(image_path, categories, output, device='cuda:0', segmenter=None):
    """Return objects.json, including explicit failed/partial outcomes for model errors."""
    image_path, output = Path(image_path).resolve(), Path(output).resolve()
    repo = Path(__file__).resolve().parents[3]
    if output.exists() or output.is_relative_to(repo):
        raise ValueError('require a fresh output directory outside the source repository')
    if not isinstance(categories, list) or not categories or any(
            not isinstance(item, str) or not item.strip() for item in categories):
        raise ValueError('categories must be a nonempty list of category strings')
    categories = list(dict.fromkeys(item.strip().lower().rstrip('.') for item in categories))
    if any(not item for item in categories):
        raise ValueError('categories must be nonempty')
    output.mkdir(parents=True)
    manifest = {'schema': SCHEMA, 'status': 'started', 'categories': categories,
                'source': {'image': str(image_path)}, 'objects': [], 'rejected': [], 'failures': [],
                'models': {'grounding_model': GROUNDING_DINO_MODEL, 'sam_model': SAM_MODEL,
                           'grounding_requested_revision': GROUNDING_REVISION,
                           'sam_requested_revision': SAM_REVISION,
                           'local_files_only': True, 'dtype': 'float32', 'device': device},
                'deduplication_mask_iou': DEDUPLICATION_IOU,
                'deduplication_mask_overlap_min': DEDUPLICATION_OVERLAP_MIN,
                'visual_quality_verified': False}
    write_json(output / 'status.json', {'schema': SCHEMA, 'status': 'started'})
    try:
        source_hash = sha256(image_path)
        manifest['source']['image_sha256'] = source_hash
        with Image.open(image_path) as opened:
            original_size = list(opened.size)
            orientation = opened.getexif().get(274)
            image = ImageOps.exif_transpose(opened).convert('RGB')
            image.load()
        image.save(output / 'source.png')
        manifest['source'].update(image_sha256=source_hash, original_size=original_size,
            input_size=list(image.size), exif_orientation=orientation,
            conversion='EXIF transpose, RGB; no resize or crop',
            canonical_image=str(output / 'source.png'), canonical_image_sha256=sha256(output / 'source.png'))
        if segmenter is None:
            segmenter = ObjectSegmenter(device)
        manifest['models'].update(getattr(segmenter, 'receipt', {'backend': 'injected'}))
        valid = []
        for category_index, category in enumerate(categories):
            try:
                candidates = segmenter(image, category)
                if not isinstance(candidates, list):
                    raise ValueError('segmenter must return a list of instances')
                if not candidates:
                    manifest['failures'].append({'category': category, 'reason': 'no_detection'})
                for index, candidate in enumerate(candidates):
                    candidate_id = f'candidate_{category_index + 1:03d}_{index + 1:03d}'
                    try:
                        valid.append(validate_candidate(candidate, category, image.size, candidate_id))
                    except (KeyError, TypeError, ValueError) as error:
                        manifest['rejected'].append({'candidate_id': candidate_id, 'category': category,
                            'reason': f'invalid_candidate:{type(error).__name__}:{error}'})
            except Exception as error:
                manifest['failures'].append({'category': category, 'reason': f'{type(error).__name__}:{error}'})
        selected = []
        # Predicted masks can cover a whole object, a part, or surrounding pixels.
        # Shared visible pixels suppress duplicate regions, not overlapping boxes.
        # This heuristic does not guarantee whole objects or exhaustive discovery.
        for candidate in sorted(valid, key=lambda row: -row['detection_score']):
            duplicate = None
            for kept in selected:
                intersection = int(np.logical_and(candidate['mask'], kept['mask']).sum())
                iou = intersection / (candidate['mask_pixels'] + kept['mask_pixels'] - intersection)
                overlap_min = intersection / min(candidate['mask_pixels'], kept['mask_pixels'])
                if iou >= DEDUPLICATION_IOU or overlap_min >= DEDUPLICATION_OVERLAP_MIN:
                    duplicate = {'reason': 'duplicate_mask', 'kept_candidate_id': kept['candidate_id'],
                                 'duplicate_criterion': 'mask_iou' if iou >= DEDUPLICATION_IOU else 'mask_overlap_min',
                                 'mask_iou': iou, 'mask_overlap_min': overlap_min,
                                 'mask_intersection_pixels': intersection, 'kept_mask_pixels': kept['mask_pixels']}
                    break
            if duplicate:
                manifest['rejected'].append({key: value for key, value in candidate.items() if key != 'mask'}
                                            | duplicate)
            else:
                selected.append(candidate)
        for index, candidate in enumerate(selected):
            manifest['objects'].append(save_object(image, candidate, output, f'object_{index + 1:03d}'))
        if sha256(image_path) != source_hash:
            raise ValueError('source image changed during segmentation')
        has_failures = manifest['failures'] or any(row['reason'] != 'duplicate_mask' for row in manifest['rejected'])
        manifest['status'] = ('partial' if has_failures else 'succeeded') if manifest['objects'] else 'failed'
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['failures'].append({'reason': f'{type(error).__name__}:{error}'})
    write_json(output / 'objects.json', manifest)
    cards = []
    for row in manifest['objects']:
        label = html.escape(row['id'] + ': ' + row['category'])
        overlay_path = Path(row['overlay']).relative_to(output).as_posix()
        mask_path = Path(row['object_mask']).relative_to(output).as_posix()
        cards.append(f'<figure><figcaption>{label}</figcaption><img src="{overlay_path}">'
                     f'<a href="{mask_path}">Binary mask</a></figure>')
    (output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Object masks</title>'
        '<style>body{font:16px sans-serif;margin:24px}figure{display:inline-block;vertical-align:top}'
        'img{display:block;max-width:480px;max-height:480px}</style>'
        f'<h1>Object masks: {manifest["status"]}</h1><p>Automatic predictions; quality not verified.</p>'
        + ''.join(cards), encoding='utf8')
    write_json(output / 'status.json', {
        'schema': SCHEMA, 'status': manifest['status'], 'objects': len(manifest['objects']),
        'manifest': str(output / 'objects.json'), 'visual_quality_verified': False,
        'failures': manifest['failures'], 'rejected': manifest['rejected'],
    })
    return output / 'objects.json'
