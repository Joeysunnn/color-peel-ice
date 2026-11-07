"""Reuse the accepted Mailbox T16 bytes; initialize with VLM, train500, infer300/500."""

import argparse
import json
import os
from pathlib import Path
import shutil

from PIL import Image

from experiments.paper_mt_transfer_v1 import run as transfer
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sdxl_mailbox_v1.data import NATURAL, NATURAL_PROTOCOL_SHA
from experiments.subject_multicolor_multimaterial_b.training import _write
from src.methods.colorpeel_ice.single_image_pipeline import ENTRY, RUNTIME, execute, fresh
from scripts.launch.colorpeel_run import git_info

EXTRACTION = NATURAL.parent / '20261018-161838__extraction__f622cc5__42'
INPUT_SHA = '1ba1e5010c6732b9cd241f6bdf7cdaee4404210b2e9e89e65d822ff713433670'
MASK_SHA = '5d9eedeffc14ff71c3afca6109bf3f3c9ad9eb9f21acfa6554f35ea432df6661'
TEXTURE_SHA = '5db55a93b23229e8830ebb22ba7fb83ffdb11aa190333fa631cfc9969c0dc838'
ROI = [335, 399, 387, 412]


def locations(object_root):
    root = Path(object_root).resolve()
    identity = read_json(root / 'object.json')
    if (root.name != transfer.OBJECT or identity['object_key'] != transfer.OBJECT
            or identity['source_image_sha256'] != transfer.contract.SOURCE_SHA):
        raise ValueError('require the existing paper Mailbox object')
    tag = '20261107__texture_legacy_t16__' + git_info()['commit'][:7] + '__42'
    return root, tag, root / 'prepared' / tag, root / 'training' / tag / 'object_003/texture'


def validate_legacy(extracted, rows, locks):
    if (extracted['selected_region'] != 'small_painted_lid'
            or extracted['texture_sampling']['crop_xyxy'] != ROI
            or extracted['texture_sampling']['no_inpainting'] is not True
            or extracted['texture_statistics']['method'] != 'linear_srgb_rec709_grayscale_robust_quantiles'
            or extracted['texture_statistics']['quantiles'] != [0.02, 0.98]
            or extracted['texture_statistics']['output_range'] != [0.25, 0.75]):
        raise ValueError('require the accepted early ROI and grayscale method')
    for name, digest in [('input.png', INPUT_SHA), ('mask.png', MASK_SHA), ('texture.png', TEXTURE_SHA)]:
        item = extracted['files'][name]
        if item['sha256'] != digest or locks.get(item['path']) != digest:
            raise ValueError('legacy input, reviewed mask or texture identity changed')
    if len(rows) != 16 or len({row['id'] for row in rows}) != 16:
        raise ValueError('require exactly the original sixteen unique rows')
    for row in rows:
        if (row['source_id'] != 'mailbox' or row['branch'] != 'texture'
                or row['concept_id'] != 'mailbox_texture' or row['prompt'].count('<T*>') != 1):
            raise ValueError('require Mailbox Texture rows and original captions')
        for field in ('image', 'mask'):
            if locks.get(row[field]) != row[field + '_sha256']:
                raise ValueError('original T16 RGB/mask bytes are not hash-locked')


def prepare(object_root, gpu):
    from experiments.natural_texture_material_v1.training_source import verify_dataset

    root, tag, output, training = locations(object_root)
    if sha256(NATURAL / 'plan/protocol.json') != NATURAL_PROTOCOL_SHA:
        raise ValueError('legacy dataset protocol changed')
    protocol, _, locks = verify_dataset(NATURAL)
    if Path(protocol['source_extraction']) != EXTRACTION:
        raise ValueError('dataset does not come from the accepted early extraction')
    extracted = read_json(EXTRACTION / 'mailbox/extraction.json')
    ledger = NATURAL / 'staging/mailbox_texture/training_assets_manifest.jsonl'
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    validate_legacy(extracted, rows, locks)
    output = fresh(output)
    references = output / 'legacy_reference'
    references.mkdir()
    for name in ('input.png', 'mask.png', 'albedo.png', 'texture.png', 'extraction.json'):
        source = EXTRACTION / 'mailbox' / name
        shutil.copyfile(source, references / name)
        if sha256(source) != sha256(references / name):
            raise ValueError('legacy reference copy changed bytes')
        locks[str(references / name)] = sha256(source)
    # This crop is only a VLM observation. The frozen texture and T16 are never recomputed.
    vlm_input = output / 'vlm_surface_rgb.png'
    with Image.open(references / 'input.png') as image:
        crop = image.convert('RGBA').crop(ROI)
        if crop.getchannel('A').getextrema() != (255, 255):
            raise ValueError('reviewed VLM surface crop contains transparency')
        crop.convert('RGB').save(vlm_input)
    command = [RUNTIME['vlm_python'], '-u', '-B', ENTRY, '--worker', 'initialize',
               '--image', vlm_input, '--output', output / 'initializers',
               '--branches', 'texture', '--gpus', str(gpu)]
    _write(output / 'vlm_command.json', [str(x) for x in command])
    execute(command, output / 'initializers.log', gpu)
    manifest = output / 'initializers/initializers.json'
    initializer = read_json(manifest)
    if initializer.get('selected_branches') != ['texture']:
        raise ValueError('VLM did not return a valid Texture initializer')
    record = {'id': 'object_003', 'category': 'mailbox', 'paper_object_key': transfer.OBJECT,
              'source_image': str(references / 'input.png'), 'source_image_sha256': INPUT_SHA,
              'object_mask': str(references / 'mask.png'), 'object_mask_sha256': MASK_SHA,
              'mask_role': 'historical reviewed material surface region, not current SAM object mask',
              'vlm_input': str(vlm_input), 'vlm_input_sha256': sha256(vlm_input),
              'vlm_input_scope': 'original RGB at the accepted early Texture crop', 'crop_box_xyxy': ROI}
    data = {'source_image_sha256': INPUT_SHA, 'source_mask_sha256': MASK_SHA,
            'source_locks': locks, 'rows': [{**row, 'caption': row['prompt']} for row in rows],
            'provenance': 'byte-identical accepted 20261018 Mailbox T16; no new extraction or rendering'}
    item = {'id': 'object_003', 'record': record, 'initializer_manifest': str(manifest),
            'datasets': {'texture': data}, 'failures': {}, 'status': 'prepared'}
    _write(output / 'object.json', item)
    request = {'status': 'prepared', 'git': git_info(), 'object_root': str(root), 'tag': tag,
               'object_manifest': str(output / 'object.json'),
               'object_manifest_sha256': sha256(output / 'object.json'),
               'training_output': str(training), 'max_steps': 500, 'checkpoint_steps': [300, 500],
               'initializer': initializer['branches']['texture']['initializer'],
               'legacy_roi': ROI, 'training_images': 16, 'old_weights_reused': False,
               'reextraction': False, 'rerendering': False, 'training_started': False,
               'inference_images': 328, 'seeds': list(transfer.SEEDS),
               'prompts': list(dict.fromkeys(row['prompt'] for row in transfer.plan('texture', 300)))}
    _write(output / 'request.json', request)
    print(json.dumps(request), flush=True)
    return output


def train_and_infer(object_root, gpu):
    root, tag, prepared, output = locations(object_root)
    request = read_json(prepared / 'request.json')
    if (request['git'] != git_info()
            or sha256(request['object_manifest']) != request['object_manifest_sha256']):
        raise ValueError('prepared recipe or producer Git changed')
    evidence = fresh(root / 'provenance' / tag)
    bundle = evidence / 'train_outputs'
    bundle.mkdir()
    command = [RUNTIME['train_python'], '-u', '-B', ENTRY, '--worker', 'train-object',
               '--prepared', request['object_manifest'], '--output', output, '--branches', 'texture',
               '--steps', 'texture=500', '--checkpoint-steps', 'texture=300,500', '--gpus', str(gpu)]
    commands = [[str(x) for x in command]]
    for step in (300, 500):
        commands.append([RUNTIME['train_python'], '-u', '-B', '-m',
                         'experiments.paper_mailbox_legacy_texture_v1.run', '--stage', 'infer',
                         '--object-root', str(root), '--step', str(step), '--gpu', str(gpu)])
    _write(bundle / 'commands.json', commands)
    (bundle / 'COMMANDS.md').write_text('\n\n'.join(json.dumps(c) for c in commands) + '\n')
    (bundle / 'SCIENTIFIC_CHANGELOG.md').write_text(
        'Reuse accepted early Mailbox ROI, q02/q98 grayscale, and all16 RGB/mask/caption rows unchanged.\n'
        'No SuperMat rerun, extraction, rendering, automatic object ROI or gain1 method.\n'
        'Fresh SD2.1 LoRA64; newest selected-v1 VLM initializer; train500, paired saves300/500.\n')
    (bundle / 'COMPARABILITY_REPORT.md').write_text(
        'Same T16 bytes and captions as the early natural Mailbox study. Base SD2.1, VLM initializer and '
        'requested dose differ from the original SD1.5 experiment. Not an initializer-only ablation.\n')
    (bundle / 'LOG.md').write_text(f'Training: {output}/training.log\nWorker: {evidence}/train_worker.log\n')
    state = {'status': 'training', 'max_steps': 500, 'completed_steps': None,
             'best_metric': None, 'best_checkpoint': None, 'stop_reason': None,
             'training_output': str(output), 'inference': {}, 'monitoring': False, 'git': git_info()}

    def save():
        _write(bundle / 'status.json', state)
        (bundle / 'SUMMARY.md').write_text(
            f"Status: {state['status']}. Training steps: {state['completed_steps']}/500. "
            f"Inference checkpoints completed: {list(state['inference'])}. No quality selection.\n")

    save()
    try:
        execute(command, evidence / 'train_worker.log', gpu)
        trained = read_json(output / 'status.json')
        if trained['status'] != 'trained' or trained['completed_steps'] != 500:
            raise ValueError('training did not finish the requested500 steps')
        link = root / 'ckpts/texture' / tag
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(output / 'ckpts', target_is_directory=True)
        state.update(status='inferring', completed_steps=500)
        save()
        for step, command in zip((300, 500), commands[1:]):
            execute(command, evidence / f'texture{step}_infer.log', gpu)
            result = root / 'infer' / f'20261107__texture{step}_transfer__{git_info()["commit"][:7]}__seeds20-60'
            status = read_json(result / 'status.json')
            if status['status'] != 'succeeded' or status['generated_images'] != 164:
                raise ValueError('requested inference grid incomplete')
            state['inference'][str(step)] = str(result)
            save()
        state.update(status='succeeded', stop_reason='requested_training_and_inference_completed')
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', stop_reason='error')
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'train-and-infer', 'infer'), required=True)
    parser.add_argument('--object-root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--step', type=int, choices=(300, 500))
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('GPU must be nonnegative')
    os.environ.update(transfer.ENVIRONMENT, CUDA_VISIBLE_DEVICES=str(args.gpu))
    if args.stage == 'prepare':
        prepare(args.object_root, args.gpu)
    elif args.stage == 'train-and-infer':
        train_and_infer(args.object_root, args.gpu)
    else:
        if args.step is None:
            parser.error('inference requires a selected checkpoint')
        _, tag, _, _ = locations(args.object_root)
        transfer.run(args.object_root, 'texture', args.step, args.gpu,
                     training_run=tag, source_git=git_info(), source_image_sha=INPUT_SHA)


if __name__ == '__main__':
    main()
