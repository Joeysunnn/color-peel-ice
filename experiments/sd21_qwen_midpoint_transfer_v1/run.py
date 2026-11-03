"""Inference-only midpoint transfer using the frozen Qwen Mailbox exports."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd21_qwen_initializers_v1 import contract
from experiments.sd21_qwen_initializers_v1.inference import (
    generation_kwargs, prompt_conditioning, register_initial_input,
)
from experiments.sd21_qwen_initializers_v1.run import ENVIRONMENT, require_idle
from experiments.sd21_wood_branches_v1 import snapshots
from experiments.subject_multicolor_multimaterial_b.training import _write
from scripts.launch.colorpeel_run import git_info, resolve_run_dir
from . import protocol

ROOT = Path(__file__).resolve().parents[2]


def validate_output(output, source):
    output, source = resolve_run_dir(Path(output)), Path(source).resolve()
    protected = (ROOT, source, contract.MODEL_DIR.resolve())
    if output.exists() or any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError('require a fresh external output separate from source, code and model')
    return output


def selected_exports(exports, manifest, audit, model):
    if set(exports) != set(contract.BRANCHES):
        raise ValueError('require exactly four source export branches')
    manifest_path = Path(audit['initializer_manifest'])
    selected = {}
    for branch, step in protocol.STEPS.items():
        item, token = exports[branch][str(step)], contract.TOKENS[branch]
        expected = {'branch': branch, 'method': 'lora_r64', 'token': token, 'step': step,
                    'steps': step, 'base_model': model, 'embedding_dim': 1024, 'status': 'verified',
                    'weight': 'pytorch_lora_kv_weights.bin', 'paired_embedding_finite': True}
        if any(item.get(key) != value for key, value in expected.items()):
            raise ValueError('selected export branch, checkpoint, base or pairing differs')
        provenance = item['initializer_provenance']
        expected = {'initializer_manifest': str(manifest_path),
                    'initializer_manifest_sha256': sha256(manifest_path),
                    'initializer': manifest['branches'][branch]['initializer'],
                    'initializer_token_id': manifest['branches'][branch]['initializer_token_id']}
        audit_file = provenance['initializer_audit']
        if (any(provenance.get(key) != value for key, value in expected.items())
                or not Path(audit_file).is_absolute()
                or audit['source_locks'].get(audit_file) != provenance['initializer_audit_sha256']):
            raise ValueError('selected initializer provenance differs from frozen source audit')
        directory = Path(item['checkpoint'])
        if not directory.is_absolute() or any(str(directory / name) not in item['hashes']
                                              for name in (item['weight'], token + '.bin')):
            raise ValueError('selected paired files must be absolute and source-locked')
        selected[branch] = item
    return selected


def _merge_locks(locks, extra):
    if any(path in locks and locks[path] != digest for path, digest in extra.items()):
        raise ValueError('conflicting source hashes')
    locks.update(extra)


def prepare(source, output):
    import torch
    from transformers import CLIPTokenizer
    from experiments.sd21_wood_branches_v1.prepare_model import verify_model
    from src.train.initializer_token_utils import load_initializer_manifest

    source = Path(source).resolve()
    if source != protocol.SOURCE.resolve():
        raise ValueError('this study requires the pinned Qwen Mailbox source experiment')
    output = validate_output(output, source)
    revision = git_info()
    paths = [source / name for name in ('source_audit.json', 'exports.json', 'status.json', 'inference/status.json')]
    audit, exports, state, inference = [read_json(path) for path in paths]
    if (audit['new_training_git'] != protocol.SOURCE_GIT or state.get('status') != 'succeeded'
            or state.get('new_training_git') != protocol.SOURCE_GIT
            or inference.get('status') != 'succeeded' or inference.get('generated_images') != 112
            or inference.get('inference_git') != protocol.SOURCE_GIT):
        raise ValueError('source experiment must have succeeded at the pinned training revision')
    locks = dict(audit['source_locks'])
    _merge_locks(locks, {str(path): sha256(path) for path in paths})
    receipt = verify_model(contract.MODEL_DIR)
    if receipt['model'] != audit['base_model']:
        raise ValueError('base model differs from source training')
    _merge_locks(locks, snapshots.model_locks(contract.MODEL_DIR, receipt))
    tokenizer = CLIPTokenizer.from_pretrained(str(contract.MODEL_DIR), subfolder='tokenizer', local_files_only=True)
    manifest_path = Path(audit['initializer_manifest'])
    if not manifest_path.is_absolute():
        raise ValueError('require an absolute initializer manifest')
    manifest = load_initializer_manifest(manifest_path, tokenizer, contract.MODEL_DIR)
    if manifest != audit['initializers']:
        raise ValueError('actual tokenizer initializer manifest differs from frozen source audit')
    _merge_locks(locks, {**manifest['source_locks'], str(manifest_path): sha256(manifest_path)})
    selected = selected_exports(exports, manifest, audit, receipt['model'])
    shapes, empty = snapshots.adapter_schema('lora_r64')
    for branch, item in selected.items():
        directory = Path(item['checkpoint'])
        if not directory.resolve().is_relative_to(source):
            raise ValueError('selected export must belong to the pinned source experiment')
        _merge_locks(locks, item['hashes'])
        snapshots.check_locks(item['hashes'])
        weights = torch.load(directory / item['weight'], map_location='cpu')
        snapshots.shared._check_state(torch, weights, shapes, empty, branch)
        if any(value.dtype != torch.float32 or not bool(torch.isfinite(value.half()).all())
               for value in weights.values()):
            raise ValueError('require FP32 adapter exports finite after FP16 conversion')
        token = item['token']
        embedding = torch.load(directory / (token + '.bin'), map_location='cpu')
        if (not isinstance(embedding, dict) or set(embedding) != {token}
                or not isinstance(embedding[token], torch.Tensor) or embedding[token].shape != (1024,)
                or embedding[token].dtype != torch.float32 or not bool(torch.isfinite(embedding[token]).all())
                or not bool(torch.isfinite(embedding[token].half()).all())):
            raise ValueError('paired learned embedding must be a finite FP32 1024-vector')
        del weights, embedding
    snapshots.check_locks(locks)
    rows = protocol.plan(manifest)
    if tokenizer.add_tokens(list(contract.TOKENS.values())) != 4 or any(
            len(tokenizer.encode(row['prompt'], add_special_tokens=True)) > 77 for row in rows):
        raise ValueError('learned tokens already exist or planned prompt exceeds CLIP77')
    if git_info() != revision:
        raise ValueError('inference checkout changed during preparation')
    output.mkdir(parents=True)
    _write(output / 'source_audit.json', {
        'study': protocol.STUDY, 'source_experiment': str(source), 'source_training_git': protocol.SOURCE_GIT,
        'inference_git': revision, 'initializers': manifest, 'initializer_manifest': str(manifest_path),
        'selected_exports': selected, 'source_locks': locks, 'base_model': receipt['model'],
        'gpu_mapping': protocol.GPUS, 'sampling': protocol.SAMPLING, 'fresh_training': False,
    })
    (output / 'generation_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
    return output, rows, revision


def gallery(output, rows, generated, state, group=None):
    records = {row['id']: row for row in generated}
    folder = output / 'groups' / group if group else output
    folder.mkdir(parents=True, exist_ok=True)
    body = [f'<h1>{html.escape(group or protocol.STUDY)}</h1>',
            f'<p>{state["status"]}: {len(generated)}/{len(rows)} images. '
            'S/M/T500; C100; seeds30–40. PNDM100 / FP16 / 512px / empty negative / safety checker. '
            'Color-only CFG6; others CFG3.5. Safety-filtered PNGs are retained. Visual quality is not automatically assessed.</p>']
    if group:
        body.append('<p><a href="../../index.html">All groups</a></p><div class="grid">')
        for row in rows:
            record = records.get(row['id'])
            body.append('<figure><figcaption>' + html.escape(f'{row["case"]} / seed{row["seed"]}') + '</figcaption>')
            if record:
                link = '../../' + record['image_path']
                body.append(f'<a href="{html.escape(link, quote=True)}"><img width="256" src="{html.escape(link, quote=True)}"></a>'
                            f'<p>{record["status"]}</p>')
            else:
                body.append('<p>pending</p>')
            body.append('<small>' + html.escape(row['prompt']) + '</small></figure>')
        body.append('</div>')
    else:
        for mode in protocol.GPUS:
            count = sum(row['mode'] == mode for row in generated)
            expected = sum(row['mode'] == mode for row in rows)
            body.append(f'<p><a href="groups/{mode}/index.html">{mode}</a>: {count}/{expected}</p>')
    (folder / 'index.html').write_text('<!doctype html><meta charset="utf-8"><style>'
        'body{font:16px sans-serif}.grid{display:flex;flex-wrap:wrap}figure{width:256px;margin:12px}'
        'img{max-width:100%}small{display:block}</style>' + ''.join(body), encoding='utf8')


def run_group(output, mode, rows, audit, gpu, *, render_gallery=gallery):
    import torch
    from experiments.sd21_wood_branches_v1 import inference as single
    from experiments.sd21_qwen_initializers_v1.attention import install_four_branch_lora
    from experiments.sd21_prompt_order_v1.conditioning import capture_initial_input, tensor_sha256

    folder = output / 'groups' / mode
    folder.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'expected_images': len(rows), 'generated_images': 0,
             'physical_gpu': gpu, 'logical_device': 'cuda:0', 'quality_verified': False}
    generated, initial_inputs, pipe = [], {}, None
    _write(folder / 'status.json', state)
    try:
        if git_info() != audit['inference_git']:
            raise ValueError('worker checkout differs from prepared inference revision')
        sources = {branch: audit['selected_exports'][branch] for branch in rows[0]['branch_steps']}
        for item in sources.values():
            snapshots.check_locks(item['hashes'])
        pipe = single.load_pipeline([] if mode == 'scmt' else list(sources.values()), 'cuda:0',
                                    model_dir=contract.MODEL_DIR, stock=False, experiment_contract=contract)
        if mode == 'scmt':
            weights = {branch: torch.load(Path(item['checkpoint']) / item['weight'], map_location='cpu')
                       for branch, item in sources.items()}
            install_four_branch_lora(pipe.unet, weights)
            del weights
            for item in sources.values():
                pipe.load_textual_inversion(item['checkpoint'], weight_name=item['token'] + '.bin')
        if {branch for branch, token in contract.TOKENS.items() if token in pipe.tokenizer.get_vocab()} != set(sources):
            raise ValueError('pipeline token inventory differs from exactly the selected branches')
        embeddings = {}
        for branch, item in sources.items():
            vector = torch.load(Path(item['checkpoint']) / (item['token'] + '.bin'), map_location='cpu')[item['token']]
            token_id = pipe.tokenizer.convert_tokens_to_ids(item['token'])
            actual = pipe.text_encoder.get_input_embeddings().weight[token_id]
            if not torch.equal(actual, vector.to(device=actual.device, dtype=actual.dtype)):
                raise ValueError('actual learned vector differs from selected paired export')
            embeddings[branch] = {'token_id': token_id, 'loaded_fp16_sha256': tensor_sha256(actual),
                                  'source_fp32_sha256': tensor_sha256(vector)}
            del actual, vector
        gates, recorded = {}, {}
        for row in rows:
            gates[row['prompt']], recorded[row['prompt']] = prompt_conditioning(pipe, row)
        _write(folder / 'token_gates.json', recorded)
        _write(folder / 'loaded_sources.json', {'sources': sources, 'loaded_embeddings': embeddings})
        pipe.set_progress_bar_config(disable=True)
        with torch.no_grad(), (folder / 'generation_status.jsonl').open('x', encoding='utf8') as ledger:
            for row in rows:
                state['active'] = row['id']
                _write(folder / 'status.json', state)
                captured = {}
                hook = capture_initial_input(pipe.unet, captured)
                try:
                    result = pipe(row['prompt'], cross_attention_kwargs=gates[row['prompt']], **generation_kwargs(row))
                finally:
                    hook.remove()
                register_initial_input(initial_inputs, row['seed'], captured)
                if (result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1
                        or len(result.images) != 1 or result.images[0].mode != 'RGB' or result.images[0].size != (512, 512)):
                    raise ValueError('require one512px RGB image with safety status')
                path = output / row['image_path']
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream:
                    result.images[0].save(stream, format='PNG')
                record = {**row, 'loaded_branches': list(sources), 'loaded_source_group': mode,
                          'initial_denoiser_input': captured, 'image_sha256': sha256(path),
                          'nsfw_content_detected': bool(result.nsfw_content_detected[0]),
                          'status': 'safety_filtered' if result.nsfw_content_detected[0] else 'ok'}
                generated.append(record)
                ledger.write(json.dumps(record) + '\n')
                ledger.flush()
                state['generated_images'] = len(generated)
                _write(folder / 'status.json', state)
        if len(generated) != len(rows) or git_info() != audit['inference_git']:
            raise ValueError('group incomplete or inference checkout changed')
        state['status'] = 'succeeded'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        del pipe
        torch.cuda.empty_cache()
        _write(folder / 'status.json', state)
        render_gallery(output, rows, generated, state, mode)


def worker(output, gpu):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise ValueError('worker CUDA visibility differs from its assigned physical GPU')
    output = Path(output).resolve()
    audit = read_json(output / 'source_audit.json')
    rows, failures = protocol.plan(audit['initializers']), {}
    for mode, assigned in protocol.GPUS.items():
        if assigned == gpu:
            try:
                run_group(output, mode, [row for row in rows if row['mode'] == mode], audit, gpu)
            except Exception as error:
                failures[mode] = f'{type(error).__name__}: {error}'
                traceback.print_exc()
    if failures:
        raise RuntimeError(f'GPU queue failures; other groups preserved: {failures}')


def collect(output, rows):
    planned, records, initial_inputs = {row['id']: row for row in rows}, {}, {}
    for mode in protocol.GPUS:
        path = output / 'groups' / mode / 'generation_status.jsonl'
        if path.exists():
            for line in path.read_text(encoding='utf8').splitlines():
                record = json.loads(line)
                if (record['id'] in records or record['id'] not in planned
                        or any(record.get(key) != value for key, value in planned[record['id']].items())
                        or record['mode'] != mode or sha256(output / record['image_path']) != record['image_sha256']
                        or record['status'] != ('safety_filtered' if record['nsfw_content_detected'] else 'ok')):
                    raise ValueError('generated image evidence differs from its plan or saved bytes')
                register_initial_input(initial_inputs, record['seed'], record['initial_denoiser_input'])
                records[record['id']] = record
    generated = [records[row['id']] for row in rows if row['id'] in records]
    (output / 'generation_status.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in generated), encoding='utf8')
    _write(output / 'initial_input_audit.json', {'same_seed_equal_across_groups': True,
                                               'generated_images': len(generated), 'by_seed': initial_inputs})
    return generated


def run(source, output):
    require_idle(set(protocol.GPUS.values()))
    output, rows, revision = prepare(source, output)
    state = {'status': 'running', 'expected_images': len(rows), 'generated_images': 0,
             'inference_git': revision, 'source_training_git': protocol.SOURCE_GIT,
             'gpu_mapping': protocol.GPUS, 'quality_verified': False, 'fresh_training': False}
    generated, failures = [], {}
    _write(output / 'status.json', state)
    for mode in protocol.GPUS:
        gallery(output, [row for row in rows if row['mode'] == mode], [], state, mode)
    gallery(output, rows, generated, state)

    def launch(gpu):
        command = [sys.executable, '-u', '-B', '-m', 'experiments.' + protocol.STUDY + '.run',
                   '--output', str(output), '--worker-gpu', str(gpu)]
        _write(output / f'gpu_{gpu}_command.json', {'command': command, 'physical_gpu': gpu})
        with (output / f'gpu_{gpu}.log').open('x', encoding='utf8') as log:
            subprocess.run(command, cwd=ROOT, env={**os.environ, **ENVIRONMENT, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        require_idle(set(protocol.GPUS.values()))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(launch, gpu): gpu for gpu in sorted(set(protocol.GPUS.values()))}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    failures[str(futures[future])] = f'{type(error).__name__}: {error}'
        generated = collect(output, rows)
        state['generated_images'] = len(generated)
        audit = read_json(output / 'source_audit.json')
        snapshots.check_locks(audit['source_locks'])
        if git_info() != revision:
            raise ValueError('inference checkout changed during generation')
        if failures or len(generated) != len(rows) or any(
                read_json(output / 'groups' / mode / 'status.json')['status'] != 'succeeded' for mode in protocol.GPUS):
            raise RuntimeError(f'inference incomplete; independent outputs preserved: {failures}')
        state['status'] = 'succeeded'
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}', worker_failures=failures)
        raise
    finally:
        _write(output / 'status.json', state)
        gallery(output, rows, generated, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=protocol.SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-gpu', type=int, choices=sorted(set(protocol.GPUS.values())), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_gpu is None:
        run(args.source, args.output)
    else:
        worker(args.output, args.worker_gpu)
