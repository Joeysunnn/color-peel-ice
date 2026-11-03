"""Read-only asset contract for the fixed external beach preservation set."""

import hashlib
import io
import json
from pathlib import Path
import random
import re

from PIL import Image


SOURCE_ROOT = Path('/home/r12user5/Documents/Jiawei/papers/ICE/outputs/preservation_images')
PROMPT = 'a photo at the beach'
SEED = 42
COUNT, SELECTED_COUNT = 1000, 200
# Canonical records from the independently downloaded 2026-11-03 inventory.
EXPECTED_INVENTORY_SHA256 = '6101d73c4387fa13ed51c80741bf93a5501c8c60cb31f799a95b6c94e2e2260b'
EVIDENCE_FILE_SHA256 = '11edbe176edbbeb46b2f106ca5f6aac77d90430fc3683fc29557efbfebe959aa'
PROMPT_EVIDENCE = ('Current ICE default and a 36-image beach-themed visual sample; '
                   'not recovered per-image generation captions.')


def inventory_digest(records):
    canonical = [{'id': row['id'], 'filename': Path(row['path']).name,
                  'sha256': row['sha256'], 'size': row['size'], 'mode': row['mode']}
                 for row in sorted(records, key=lambda row: row['id'])]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prepare(source_dir, output_manifest):
    """Verify all original images, then register one image per step for 200 steps.

    Returns absolute manifest/inventory paths and path-to-SHA256 source_locks.
    No image is copied or modified; both output JSON files must be new.
    """
    source, output = Path(source_dir).resolve(), Path(output_manifest).resolve()
    inventory_path = output.parent / 'externalinventory.json'
    if source != SOURCE_ROOT.resolve():
        raise ValueError('only the authorized preservation_images source is allowed')
    if output.is_relative_to(source) or output == inventory_path:
        raise ValueError('manifest must be outside the image source and distinct from the inventory')
    if output.exists() or inventory_path.exists():
        raise ValueError('manifest and inventory outputs must be new')

    indexed = {}
    for path in source.iterdir():
        match = re.fullmatch(r'(\d+)-.+\.jpg', path.name)
        if not path.is_file() or path.is_symlink() or match is None:
            raise ValueError(f'unexpected preservation source entry: {path.name}')
        image_id = int(match.group(1))
        if image_id in indexed:
            raise ValueError(f'duplicate preservation image id: {image_id}')
        indexed[image_id] = path
    if sorted(indexed) != list(range(COUNT)):
        raise ValueError('require exactly one preservation image for every id 0..999')

    records = []
    for image_id in range(COUNT):
        path = indexed[image_id]
        payload = path.read_bytes()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if image.format != 'JPEG' or image.size != (512, 512) or image.mode != 'RGB':
                raise ValueError(f'preservation image must be JPEG 512x512 RGB: {path.name}')
        records.append({'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest(),
                        'size': [512, 512], 'mode': 'RGB', 'id': image_id})
    digest = inventory_digest(records)
    if digest != EXPECTED_INVENTORY_SHA256:
        raise ValueError('preservation inventory differs from the pre-registered source evidence')

    inventory = {'schema': 1, 'source_root': str(source), 'count': COUNT,
                 'canonical_sha256': digest, 'evidence_file_sha256': EVIDENCE_FILE_SHA256,
                 'generation_model': None, 'generation_seed': None, 'records': records}
    manifest = {'schema': 1, 'source_root': str(source), 'prompt': PROMPT, 'seed': SEED,
                'prompt_evidence': PROMPT_EVIDENCE, 'generation_model': None, 'generation_seed': None,
                'sampling': 'random.Random(42).sample(numeric_id_sorted_records, 200)',
                'inventory_path': str(inventory_path), 'inventory_canonical_sha256': digest,
                'records': random.Random(SEED).sample(records, SELECTED_COUNT)}
    output.parent.mkdir(parents=True, exist_ok=True)
    locks = {row['path']: row['sha256'] for row in records}
    for path, document in ((inventory_path, inventory), (output, manifest)):
        payload = (json.dumps(document, indent=2) + '\n').encode()
        with path.open('xb') as handle:
            handle.write(payload)
        locks[str(path)] = hashlib.sha256(payload).hexdigest()
    return {'manifest_path': str(output), 'inventory_path': str(inventory_path),
            'source_locks': locks, 'count': SELECTED_COUNT, 'inventory_count': COUNT,
            'inventory_canonical_sha256': digest}
