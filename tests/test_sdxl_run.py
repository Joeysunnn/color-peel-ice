"""Validate independent SDXL queues and immutable model provenance without GPUs."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.sdxl_mailbox_v1 import run
from experiments.sdxl_mailbox_v1 import prepare_model as model


class QueueTests(unittest.TestCase):
    def jobs(self, root, smoke=False):
        sources = {branch: {'manifest': str(root / branch / 'data.json')}
                   for branch in run.contract.TRAINING}
        return run.make_jobs(root, sources, root / 'model', (1, 2, 3), smoke)

    def test_matrix_preserves_both_methods_and_pairs_the_training_directory(self):
        jobs = self.jobs(Path('/new/experiment'))
        self.assertEqual(len(jobs), 8)
        self.assertEqual({(job['branch'], job['method']) for job in jobs},
                         {(branch, method) for branch in run.contract.TRAINING
                          for method in ('fullrank', 'lora_r64')})
        total = 0
        for job in jobs:
            config = job['config']
            self.assertEqual(config['inference_seeds'], list(range(30, 41)))
            self.assertEqual(config['resolution'], 1024)
            self.assertEqual(config['steps'], 150 if job['branch'] == 'color' else 1000)
            self.assertEqual(config['checkpoints'], [100, 150] if job['branch'] == 'color'
                             else [400, 600, 800, 1000])
            command = job['inference_command']
            self.assertEqual(command[command.index('--training') + 1],
                             job['training_command'][-1])
            self.assertNotIn('composition', ' '.join(command))
            total += len(run.contract.transfer_cases(job['branch'], config['checkpoints']))
        self.assertEqual(total, 3344)

    def test_smoke_is_eight_paired_two_update_checks(self):
        jobs = self.jobs(Path('/new/smoke'), smoke=True)
        for job in jobs:
            self.assertEqual(job['config']['steps'], 2)
            self.assertEqual(job['config']['checkpoints'], [2])
            self.assertEqual(len(run.contract.transfer_cases(job['branch'], [2], smoke=True)), 1)

    def test_training_failure_prevents_inference_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = self.jobs(root)[0]
            Path(job['group']).mkdir(parents=True)
            with patch.object(run, 'git_revision', return_value={'commit': '123'}), \
                    patch.object(run.data, 'load_rows', return_value=[]), \
                    patch.object(run.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'train')) as launch:
                with self.assertRaises(subprocess.CalledProcessError):
                    run.run_queue(1, [job], root, {'commit': '123'})
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(json.loads((root / 'gpu_1.json').read_text())['status'], 'failed')
            state = json.loads((Path(job['group']) / 'status.json').read_text())
            self.assertEqual(state['completed_steps'], 0)
            self.assertEqual(state['status'], 'failed')

    def test_occupied_gpu_is_rejected(self):
        with patch.object(run.subprocess, 'check_output', return_value='0, 599\n1, 9000\n2, 9\n'):
            with self.assertRaises(RuntimeError):
                run.check_gpus((1, 2))
            run.check_gpus((0, 2))

    def test_written_configs_lock_git_model_receipt_and_dataset_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / 'model'
            model_dir.mkdir()
            (model_dir / 'model_source.json').write_text('{"revision":"test"}')

            def prepare_data(output):
                sources = {}
                for branch in run.contract.TRAINING:
                    manifest = output / branch / 'manifest.json'
                    manifest.parent.mkdir(parents=True)
                    manifest.write_text('[]')
                    sources[branch] = {'manifest': str(manifest)}
                return sources

            with patch.object(run, 'git_revision', return_value={'commit': 'verified-commit'}), \
                    patch.object(run, 'verify_model', return_value={'revision': 'test'}), \
                    patch.object(run.data, 'prepare_data', side_effect=prepare_data):
                state = run.run(root / 'experiment', model_dir=model_dir)
            self.assertEqual(state['status'], 'dry_run_verified')
            for job in state['jobs']:
                config = json.loads(Path(job['config_path']).read_text())
                self.assertEqual(config['git_commit'], 'verified-commit')
                self.assertEqual(config['model_source_sha256'], model.sha256(model_dir / 'model_source.json'))
                self.assertEqual(config['dataset_manifest_sha256'], model.sha256(config['dataset']))


class ModelTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'base' / model.REVISION
        checker = root / 'checker' / model.CHECKER_REVISION
        values = {
            'model_index.json': {'_class_name': 'StableDiffusionXLPipeline'},
            'unet/config.json': {'cross_attention_dim': 2048, 'sample_size': 128,
                                 'addition_embed_type': 'text_time', 'addition_time_embed_dim': 256,
                                 'projection_class_embeddings_input_dim': 2816},
            'text_encoder/config.json': {'hidden_size': 768, 'max_position_embeddings': 77},
            'text_encoder_2/config.json': {'hidden_size': 1280, 'projection_dim': 1280,
                                           'max_position_embeddings': 77},
            'vae/config.json': {'scaling_factor': 0.13025},
            'scheduler/scheduler_config.json': {'prediction_type': 'epsilon'},
        }
        for component, names in model.REQUIRED.items():
            origin = checker if component in ('safety_checker', 'feature_extractor') else source
            for name in names:
                path = origin / component / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
        for relative, value in values.items():
            (source / relative).write_text(json.dumps(value))
        (checker / 'safety_checker/pytorch_model.bin').write_bytes(b'test weights')
        return source, checker

    def test_receipt_locks_both_encoders_and_separate_checker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, checker = self.fixture(root)
            receipt = model.describe(source, checker, root / 'bundle')
            self.assertEqual(receipt['revision'], model.REVISION)
            self.assertIn(str(source / 'text_encoder_2/model.safetensors'), receipt['source_locks'])
            self.assertIn(str(checker / 'safety_checker/pytorch_model.bin'), receipt['source_locks'])
            self.assertEqual(receipt['checker_revision'], model.CHECKER_REVISION)
            previous = receipt['source_locks'][str(source / 'text_encoder_2/model.safetensors')]
            (source / 'text_encoder_2/model.safetensors').write_bytes(b'changed')
            current = model.describe(source, checker, root / 'bundle')
            self.assertNotEqual(previous, current['source_locks'][str(source / 'text_encoder_2/model.safetensors')])

    def test_refiner_or_wrong_base_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, checker = self.fixture(root)
            (source / 'unet/config.json').write_text(json.dumps({'cross_attention_dim': 1280}))
            with self.assertRaises(ValueError):
                model.describe(source, checker, root / 'bundle')


if __name__ == '__main__':
    unittest.main()
