"""Freeze the cached official SDXL 1.0 base and the established common checker."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPO = 'stabilityai/stable-diffusion-xl-base-1.0'
REVISION = '462165984030d82259a11f4367a4eed129e94a7b'
SOURCE_DIR = Path('/var/tmp/colorpeel-r12user5/assets/marble_c_v1/hf_cache') / (
    'models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/' + REVISION)
CHECKER_REVISION = '133a221b8aa7292a167afc5127cb63fb5005638b'
CHECKER_DIR = Path('/home/r12user5/.cache/huggingface/hub') / (
    'models--CompVis--stable-diffusion-v1-4/snapshots/' + CHECKER_REVISION)
MODEL_DIR = Path('/home/r12user5/Documents/Jiawei/colorpeel-assets') / (
    'sdxl_base_' + REVISION[:7] + '_common_checker')
COMPONENTS = ('unet', 'vae', 'text_encoder', 'text_encoder_2', 'tokenizer',
              'tokenizer_2', 'scheduler', 'safety_checker', 'feature_extractor')
REQUIRED = {
    'unet': ('config.json', 'diffusion_pytorch_model.safetensors'),
    'vae': ('config.json', 'diffusion_pytorch_model.safetensors'),
    'text_encoder': ('config.json', 'model.safetensors'),
    'text_encoder_2': ('config.json', 'model.safetensors'),
    'tokenizer': ('vocab.json', 'merges.txt', 'tokenizer_config.json', 'special_tokens_map.json'),
    'tokenizer_2': ('vocab.json', 'merges.txt', 'tokenizer_config.json', 'special_tokens_map.json'),
    'scheduler': ('scheduler_config.json',),
    'safety_checker': ('config.json',),
    'feature_extractor': ('preprocessor_config.json',),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def describe(source, checker, output):
    if source.name != REVISION or checker.name != CHECKER_REVISION:
        raise ValueError('require the pinned official model and checker snapshot revisions')
    index = read(source / 'model_index.json')
    unet = read(source / 'unet/config.json')
    encoders = [read(source / name / 'config.json') for name in ('text_encoder', 'text_encoder_2')]
    vae = read(source / 'vae/config.json')
    scheduler = read(source / 'scheduler/scheduler_config.json')
    if (index.get('_class_name') != 'StableDiffusionXLPipeline'
            or unet.get('cross_attention_dim') != 2048 or unet.get('sample_size') != 128
            or unet.get('addition_embed_type') != 'text_time'
            or unet.get('addition_time_embed_dim') != 256
            or unet.get('projection_class_embeddings_input_dim') != 2816
            or [item.get('hidden_size') for item in encoders] != [768, 1280]
            or encoders[1].get('projection_dim') != 1280
            or any(item.get('max_position_embeddings') != 77 for item in encoders)
            or scheduler.get('prediction_type') != 'epsilon'
            or vae.get('scaling_factor') != 0.13025):
        raise ValueError('source architecture differs from SDXL 1.0 base')
    files = {}
    for component in COMPONENTS:
        base = checker if component in ('safety_checker', 'feature_extractor') else source
        folder = base / component
        for name in REQUIRED[component]:
            if not (folder / name).is_file() or (folder / name).stat().st_size == 0:
                raise FileNotFoundError(folder / name)
        if component == 'safety_checker' and not any(
                (folder / name).is_file() for name in ('pytorch_model.bin', 'model.safetensors')):
            raise FileNotFoundError('common checker weights missing')
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                files[str(path)] = sha256(path)
    files[str(source / 'model_index.json')] = sha256(source / 'model_index.json')
    return {'repo': REPO, 'revision': REVISION, 'root': str(output), 'source_root': str(source),
            'checker_root': str(checker), 'checker_repo': 'CompVis/stable-diffusion-v1-4',
            'checker_revision': CHECKER_REVISION, 'source_locks': files,
            'architecture': {'cross_attention_dim': 2048, 'text_hidden_sizes': [768, 1280],
                             'pooled_dim': 1280, 'time_ids': 6, 'resolution': 1024},
            'note': 'SDXL base only; no refiner. Native SDXL index retained. The common SD1.4 '
                    'checker is called explicitly after SDXL decoding; it is not a native SDXL component.'}


def verify_model(path=MODEL_DIR):
    path = Path(path).resolve()
    receipt = read(path / 'model_source.json')
    source, checker = (Path(receipt[key]).resolve() for key in ('source_root', 'checker_root'))
    if describe(source, checker, path) != receipt:
        raise ValueError('model source hashes or architecture changed')
    for component in COMPONENTS:
        origin = checker if component in ('safety_checker', 'feature_extractor') else source
        if (path / component).resolve() != (origin / component).resolve():
            raise ValueError('model component link differs: ' + component)
    if sha256(path / 'model_index.json') != receipt['source_locks'][str(source / 'model_index.json')]:
        raise ValueError('SDXL model index changed')
    return receipt


def prepare(output=MODEL_DIR, source=SOURCE_DIR, checker=CHECKER_DIR):
    output, source, checker = (Path(value).resolve() for value in (output, source, checker))
    if (output.is_relative_to(ROOT) or output.is_relative_to(source) or source.is_relative_to(output)
            or output.is_relative_to(checker) or checker.is_relative_to(output)):
        raise ValueError('use an external model bundle separate from original sources')
    if output.exists():
        receipt = verify_model(output)
        if receipt['source_root'] != str(source) or receipt['checker_root'] != str(checker):
            raise ValueError('existing bundle references other source directories')
        return receipt
    receipt = describe(source, checker, output)
    output.mkdir(parents=True)
    for component in COMPONENTS:
        origin = checker if component in ('safety_checker', 'feature_extractor') else source
        (output / component).symlink_to(origin / component, target_is_directory=True)
    (output / 'model_index.json').symlink_to(source / 'model_index.json')
    (output / 'model_source.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=MODEL_DIR)
    parser.add_argument('--source', type=Path, default=SOURCE_DIR)
    parser.add_argument('--checker', type=Path, default=CHECKER_DIR)
    args = parser.parse_args()
    result = prepare(args.output, args.source, args.checker)
    print(json.dumps({'status': 'verified', 'root': result['root'], 'revision': result['revision'],
                      'source_file_count': len(result['source_locks'])}, indent=2))
