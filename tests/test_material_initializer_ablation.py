"""The initializer intervention must not silently change training or inference."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.sd21_material_initializer_ablation_v1 import run
from experiments.sd21_qwen_material250_v1 import run as material


class AblationTests(unittest.TestCase):
    def configs(self, root):
        original = {'run': {'study': 'old', 'variant': 'old', 'seed': 42},
            'environment': {'CUDA_VISIBLE_DEVICES': '2', 'HF_HUB_OFFLINE': '1'},
            'args': {'initializer_token': 'painted', 'initializer_manifest': 'old.json',
                     'max_train_steps': 1000, 'checkpoint_steps': [500, 1000], 'seed': 42,
                     'learning_rate': 1e-5, 'kv_learning_rate': 5e-5, 'concepts_list': 'fixed8.json'},
            'data_lineage': {'initializer_manifest': 'old.json', 'rows': 8},
            'data_manifest': 'same_images_captions_masks.jsonl'}
        configs = {}
        for word in run.WORDS:
            path = root / (word + '.json')
            path.write_text(json.dumps({'branches': {'material': {'initializer': word}}}), encoding='utf8')
            with patch.object(material.contract, 'build_config', side_effect=lambda *a: copy.deepcopy(original)):
                config = material.build_config(material.SOURCE, run.GPUS[word], run.STEPS, initializer_manifest=path)
                recipe = material.recipe(material.SOURCE, run.GPUS[word], run.STEPS, initializer_manifest=path)
                self.assertEqual(recipe.build_config('material', 'lora_r64'), config)
            configs[word] = config
        return configs

    def test_only_initializer_provenance_identity_gpu_differ_and_export_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            configs = self.configs(Path(temporary))
            run.verify_comparability(configs)
            for word, config in configs.items():
                self.assertEqual(config['args']['concepts_list'], 'fixed8.json')
                self.assertEqual(config['args']['initializer_token'], word)
                self.assertEqual(config['args']['seed'], 42)
                self.assertEqual(config['args']['learning_rate'], 1e-5)
                self.assertEqual(config['args']['kv_learning_rate'], 5e-5)
                self.assertEqual(config['data_lineage']['rows'], 8)

    def test_comparability_rejects_lr_data_seed_checkpoint_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            configs = self.configs(Path(temporary))
        for key, value in (('learning_rate', 2e-5), ('concepts_list', 'other.json'),
                           ('seed', 43), ('checkpoint_steps', [200])):
            changed = copy.deepcopy(configs)
            changed['glossy']['args'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                run.verify_comparability(changed)

    def test264_images_have_same_prompts_sampling_and_seeds_and_separate_paths(self):
        paths, baseline = set(), None
        for word in run.WORDS:
            manifest = {'subject_category': 'mailbox', 'branches': {branch: {
                'token': token, 'initializer': word if branch == 'material' else branch}
                for branch, token in material.contract.TOKENS.items()}}
            for step in run.STEPS:
                rows = material.plan(manifest, step)
                self.assertEqual(len(rows), 44)
                observed = [(row['prompt'], row['seed'], row['guidance_scale'], row['negative_prompt']) for row in rows]
                baseline = observed if baseline is None else baseline
                self.assertEqual(observed, baseline)
                for row in rows:
                    self.assertEqual(row['mode'], 'material_only')
                    self.assertEqual(row['branch_steps'], {'material': step})
                    self.assertEqual(row['guidance_scale'], 3.5)
                    self.assertEqual(row['negative_prompt'], '')
                    self.assertEqual(row['prompt'], f'a photo of a {row["case"]} with a <M*> surface')
                    paths.add(f'{word}/inference/step{step}/{row["image_path"]}')
                self.assertEqual({r['case'] for r in rows}, {'chair', 'vase', 'sofa', 'mug'})
                self.assertEqual({r['seed'] for r in rows}, set(range(30, 41)))
        self.assertEqual(len(paths), 264)


if __name__ == '__main__':
    unittest.main()
