"""Train original and rank64 on the approved27 images, then run four Subject probes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import html
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.sd15_token_local_kv_v1.contract import sha256
from experiments.sd21_wood_branches_v1 import run as shared
from experiments.sd21_wood_branches_v1.prepare_model import verify_model
from experiments.sd21_wood_subject27_v1 import contract
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info, read_config, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(30, 61))
REQUEST = ('用这27张数据在SD2.1上重训wood spoon subject，original和rank64都做，沿用之前设置，'
           '各1000步，保存400/600/800/1000步并做相同的transfer测试，生成seed30–60。'
           '放同一个新实验文件夹，严格本地修改提交推送、服务器拉取运行，旧结果不要动，启动后不用监控。')


def jobs(output, revision, gpus):
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    result = []
    for number, method in enumerate(contract.METHODS):
        group = output / 'methods' / method
        config = contract.build_config('subject', method)
        name = f"{stamp}__{contract.STUDY}__{config['run']['variant']}__{revision['commit'][:7]}__42"
        training_run, config_path = group / 'training' / name, group / 'config.json'
        result.append({'branch': 'subject', 'method': method, 'gpu': gpus[number % len(gpus)],
            'group': str(group), 'run': str(training_run), 'config': config, 'config_path': str(config_path),
            'training_command': [sys.executable, '-B', str(ROOT / 'scripts/launch/colorpeel_run.py'),
                                 '--config', str(config_path), '--run-dir', str(training_run)],
            'inference_command': [sys.executable, '-u', '-B', '-m',
                'experiments.sd21_wood_subject27_v1.inference', '--run', str(training_run),
                '--output', str(group / 'inference'), '--method', method, '--execute']})
    return result


def report(group, state, job):
    shared.report_training(group, state, job)
    folder = group / 'train_outputs'
    (folder / 'SCIENTIFIC_CHANGELOG.md').write_text(
        'Same pinned SD2.1-base512 and original/rank64 token-local K/V methods. '
        'Only the dataset changes from72 to the approved27 RGB/mask/caption pairs. '
        'All other science arguments, initialization, losses, optimizer, learning '
        'rates, seed42, float32 training and fixed sample order are inherited. '
        'No resume or old learned weights.1000 updates, snapshots400/600/800/1000.\n', encoding='utf8')
    (folder / 'COMPARABILITY_REPORT.md').write_text(
        'The27-image input is24 blue/green studio M/R endpoints plus the original '
        'wood photograph and its two LCh recolors. Source metadata remain read-only; '
        'the new authorization.json records this training approval. '
        'At1000 updates the first frozen row is seen38 times and the other26 rows37 '
        'times; the72-image baseline had13/14 exposures. This changes both data and '
        'per-image exposure. Keep the established nine independent Subject probes, '
        'seeds30–60,PNDM100/CFG3.5/512px/fp16/common checker. '
        'Same-step embeddings and adapters are paired, with ordinary embedding drift checked. '
        'Startup verification does not establish output quality.\n', encoding='utf8')


def input_review(output, data):
    shutil.copytree(contract.DATASET, output / 'inputs')
    cards = []
    for row in data['references']:
        for field, folder in (('image', 'images'), ('mask', 'masks')):
            copied = output / 'inputs/assets' / row['id'] / folder / 'image.png'
            if sha256(copied) != row[field + '_sha256']:
                raise ValueError('input review copy differs from the actual training source')
        path = f'inputs/assets/{row["id"]}/images/image.png'
        cards.append(f'<figure><img width="256" src="{path}"><figcaption>{html.escape(row["caption"])}</figcaption></figure>')
    (output / 'input_review.html').write_text('<!doctype html><meta charset="utf-8">'
        '<h1>27 actual training RGB images and captions</h1>'
        '<p>Byte-identical review copies. Training reads the frozen source paths recorded in authorization.json.</p>'
        '<section style="display:flex;flex-wrap:wrap;gap:12px">' + ''.join(cards) + '</section>', encoding='utf8')


def run(output, gpus=(2, 3), execute=False):
    output = resolve_run_dir(Path(output))
    revision = git_info()
    if (output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(contract.SOURCE_ROOT)
            or not gpus or len(set(gpus)) != len(gpus) or any(type(gpu) is not int or gpu < 0 for gpu in gpus)):
        raise ValueError('use a fresh external experiment, separate from the dataset, and distinct GPUs')
    data, model = contract.verify_dataset('subject'), verify_model()
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 24 * 1024 ** 3:
        raise OSError('require24GiB free for eight complete training snapshots and outputs')
    if execute:
        stats = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                         '--format=csv,noheader,nounits'], text=True)
        used = {int(row.split(',')[0]): int(row.split(',')[1]) for row in stats.splitlines()}
        if any(used.get(gpu, 999999) > 2048 for gpu in gpus):
            raise RuntimeError('selected GPU is occupied; preserve existing jobs')
    planned = jobs(output, revision, gpus)
    output.mkdir()
    input_review(output, data)
    _write(output / 'authorization.json', {'request': REQUEST, 'git': revision, 'training_authorized': True,
        'source': str(contract.SOURCE_ROOT), 'source_audit_sha256': contract.AUDIT_SHA256,
        'source_metadata_unchanged': True, 'historical_dataset_only_flags_retained': True,
        'methods': list(contract.METHODS), 'steps': 1000, 'checkpoint_steps': list(contract.STEPS),
        'generation_seeds': list(SEEDS), 'monitoring': False})
    for job in planned:
        Path(job['group']).mkdir(parents=True)
        _write(Path(job['config_path']), job['config'])
        if read_config(Path(job['config_path'])) != job['config']:
            raise ValueError('launcher config roundtrip changed the training recipe')
    state = {'status': 'running' if execute else 'dry_run_verified', 'git': revision,
             'study': contract.STUDY, 'data': data, 'model': model, 'jobs': planned,
             'training_images': 27, 'max_train_steps': 1000, 'checkpoint_steps': list(contract.STEPS),
             'generation_seeds': list(SEEDS), 'expected_images': 2232, 'monitoring': False}
    _write(output / 'experiment.json', state)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>SD2.1 wood Subject27</h1>'
        '<p>Original and rank64;1000updates;400/600/800/1000;9probes ×seeds30–60.</p>'
        '<p><a href="input_review.html">Actual27 RGB and captions</a></p>'
        + ''.join(f'<p><a href="methods/{method}/inference/index.html">{method}: all checkpoints</a></p>'
                  for method in contract.METHODS), encoding='utf8')
    if not execute:
        return state
    try:
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(shared.run_queue, gpu, [job for job in planned if job['gpu'] == gpu],
                                   output, revision, dataset_verifier=contract.verify_dataset,
                                   training_reporter=report) for gpu in gpus]
            state['queues'] = [future.result() for future in futures]
        state['status'] = 'succeeded'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        _write(output / 'experiment.json', state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpus', nargs='+', type=int, default=[2, 3])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run(args.output, tuple(args.gpus), args.execute)
