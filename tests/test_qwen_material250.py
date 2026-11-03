"""Material250 must keep the existing scientific recipe and isolate its inference."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.sd21_qwen_initializers_v1 import contract
from experiments.sd21_qwen_material250_v1 import run


class Material250Tests(unittest.TestCase):
    def test150_200_is_one200_step_training_with_both_paired_snapshots(self):
        original = {'run': {'study': 'old', 'variant': 'material_fresh', 'seed': 42},
                    'environment': {'CUDA_VISIBLE_DEVICES': '2'},
                    'args': {'max_train_steps': 1000, 'checkpoint_steps': [500, 1000],
                             'lr_scheduler': 'constant', 'lr_warmup_steps': 0}}
        with patch.object(contract, 'build_config', side_effect=lambda *a: copy.deepcopy(original)):
            config = run.build_config(run.SOURCE, 0, (150, 200))
            recipe = run.recipe(run.SOURCE, 0, (150, 200))
            self.assertEqual(recipe.build_config('material', 'lora_r64'), config)
        self.assertEqual(recipe.STEPS, (150, 200))
        self.assertEqual(config['args']['max_train_steps'], 200)
        self.assertEqual(config['args']['checkpoint_steps'], [150, 200])
        self.assertEqual(config['run'], {'study': 'sd21_qwen_material150_200_v1',
                                        'variant': 'material_fresh200', 'seed': 42})
        self.assertEqual(config['args']['lr_scheduler'], 'constant')
        self.assertEqual(config['args']['lr_warmup_steps'], 0)

    def test150_200_output_paths_ids_and_gallery_labels_remain_separate(self):
        words = {'subject': 'mailbox', 'color': 'orange', 'material': 'painted', 'texture': 'uniform'}
        manifest = {'subject_category': 'mailbox', 'branches': {
            b: {'token': contract.TOKENS[b], 'initializer': word} for b, word in words.items()}}
        paths, ids = set(), set()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for step in (150, 200):
                rows = run.plan(manifest, step)
                folder = run.inference_dir(output, (150, 200), step)
                self.assertEqual(len(rows), 44)
                for row in rows:
                    self.assertEqual(row['branch_steps'], {'material': step})
                    self.assertEqual(row['phase'], f'step{step}')
                    paths.add(folder / row['image_path'])
                    ids.add(row['id'])
                run.gallery(folder, rows, [], {'status': 'waiting_for_training'})
                page = (folder / 'groups/material_only/index.html').read_text(encoding='utf8')
                self.assertIn(f'Material{step} only', page)
                self.assertNotIn('Material250', page)
            self.assertEqual(run.inference_dir(output, (250,), 250), output / 'inference')
        self.assertEqual(len(paths), 88)
        self.assertEqual(len(ids), 88)

    def test_config_changes_only_horizon_run_identity_and_gpu(self):
        original = {'run': {'study': 'old', 'variant': 'material_fresh', 'seed': 42},
                    'environment': {'CUDA_VISIBLE_DEVICES': '2', 'HF_HUB_OFFLINE': '1'},
                    'args': {'max_train_steps': 1000, 'checkpoint_steps': [500, 1000],
                             'lr_scheduler': 'constant', 'lr_warmup_steps': 0,
                             'learning_rate': 1e-5, 'kv_learning_rate': 5e-5,
                             'initializer_token': 'painted'},
                    'data_manifest': 'unchanged-material-data'}
        with patch.object(contract, 'build_config', return_value=copy.deepcopy(original)):
            result = run.build_config(run.SOURCE, 0)
        self.assertEqual(result['args']['max_train_steps'], 250)
        self.assertEqual(result['args']['checkpoint_steps'], [250])
        self.assertEqual(result['run']['seed'], 42)
        self.assertEqual(result['environment']['CUDA_VISIBLE_DEVICES'], '0')
        result['run'] = original['run']
        result['environment']['CUDA_VISIBLE_DEVICES'] = '2'
        result['args']['max_train_steps'] = 1000
        result['args']['checkpoint_steps'] = [500, 1000]
        self.assertEqual(result, original)

    def test_44_material_only_rows_use250_and_all11_seeds(self):
        words = {'subject': 'mailbox', 'color': 'orange', 'material': 'painted', 'texture': 'uniform'}
        manifest = {'subject_category': 'mailbox', 'branches': {
            b: {'token': contract.TOKENS[b], 'initializer': word} for b, word in words.items()}}
        rows = run.plan(manifest)
        self.assertEqual(len(rows), 44)
        self.assertEqual(len({r['id'] for r in rows}), 44)
        self.assertEqual({r['case'] for r in rows}, {'chair', 'vase', 'sofa', 'mug'})
        for case in ('chair', 'vase', 'sofa', 'mug'):
            self.assertEqual({r['seed'] for r in rows if r['case'] == case}, set(range(30, 41)))
        for row in rows:
            self.assertEqual(row['branch_steps'], {'material': 250})
            self.assertEqual(row['mode'], 'material_only')
            self.assertEqual(row['phase'], 'step250')
            self.assertEqual(row['guidance_scale'], 3.5)
            self.assertEqual(row['prompt'], f'a photo of a {row["case"]} with a <M*> surface')


if __name__ == '__main__':
    unittest.main()
