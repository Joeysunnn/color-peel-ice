"""Exercise prompt scope, paired source guards and real dual-adapter dispatch without CUDA."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from experiments.mailbox_color_sd15_v1 import inference, prompts


class Array(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def to(self, device):
        return self


class Tokenizer:
    model_max_length = 77

    def convert_tokens_to_ids(self, token):
        return {'<S*>': 1, '<C*>': 2}[token]

    def __call__(self, prompt, **kwargs):
        ids = [self.convert_tokens_to_ids(word) if word in ('<S*>', '<C*>') else 3 for word in prompt.split()]
        return SimpleNamespace(input_ids=Array([ids + [0] * (77 - len(ids))]))


class MailboxColorInferenceTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix='mailbox-color-infer-', dir=os.environ.get('TMP'))
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        environment = patch.dict(os.environ, {'COLORPEEL_OUTPUT_ROOT': str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)

    def test_exact_scope_matches_old_text_and_all_inclusive_seeds(self):
        old = json.loads(prompts.SOURCE.read_text())['prompts']
        protocol = json.loads((prompts.ROOT / 'experiments/mailbox_color_sd15_v1/protocol.json').read_text())
        self.assertEqual(list(prompts.STEPS), protocol['steps'])
        self.assertEqual(list(prompts.SEEDS), protocol['seeds'])
        self.assertEqual(list(prompts.JOINT_PROMPTS), protocol['joint_prompts'])
        self.assertEqual(prompts.SOURCE.relative_to(prompts.ROOT).as_posix(), protocol['transfer_prompt_source'])
        self.assertTrue(all(prompts.SAMPLING[key] == value for key, value in protocol['sampling'].items()))
        all_rows = [row for method in prompts.METHODS for step in prompts.STEPS for row in prompts.prompt_rows(method, step)]
        self.assertEqual(len(all_rows), 3224)
        self.assertEqual(len({row['id'] for row in all_rows}), 3224)
        rows = prompts.prompt_rows('fullrank', 400)
        self.assertEqual({row['seed'] for row in rows}, set(range(30, 61)))
        self.assertEqual([row['prompt'] for row in rows[::31]][:10], [row['prompt'] for row in old])
        self.assertEqual(sum(row['arm'] == 'subject_color' for row in rows), 93)
        self.assertEqual(prompts.SAMPLING['guidance_scale'], 6.0)
        for method, step in (('wrong', 400), ('fullrank', True), ('fullrank', 500)):
            with self.assertRaises(ValueError):
                prompts.prompt_rows(method, step)

    def modules(self):
        self.calls, self.loads, self.installs = [], [], []
        owner = self
        class Scheduler:
            pass
        class UNet:
            device = 'cpu'
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()
            def load_attn_procs(self, *args, **kwargs):
                owner.installs.append(('fullrank', args, kwargs))
        class Pipe:
            def __init__(self, **kwargs):
                self.unet, self.tokenizer = kwargs['unet'], Tokenizer()
                self.safety_checker, self.scheduler, self.tokens = object(), Scheduler(), []
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls(**kwargs)
            def to(self, device):
                return self
            def load_textual_inversion(self, checkpoint, weight_name):
                self.tokens.append(weight_name)
            def __call__(self, prompt, **kwargs):
                owner.calls.append((tuple(self.tokens), prompt, kwargs))
                return SimpleNamespace(images=[Image.new('RGB', (512, 512), 'orange')], nsfw_content_detected=[False])
        generator = lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: seed)
        torch = SimpleNamespace(float16='fp16', load=lambda path, **kwargs: str(path),
            Generator=generator, zeros_like=lambda item: Array(np.zeros_like(item)),
            cat=lambda items: Array(np.concatenate(items)), cuda=SimpleNamespace(empty_cache=lambda: None))
        return {'torch': torch, 'diffusers': SimpleNamespace(DiffusionPipeline=Pipe, PNDMScheduler=Scheduler),
            'custom_attention.unet_2d_condition_custom': SimpleNamespace(UNet2DConditionModel=UNet),
            'src.methods.colorpeel_ice.dual_token_local_kv': SimpleNamespace(install_dual_token_local_kv=
                lambda *args, **kwargs: owner.installs.append(('dual_fullrank', args, kwargs))),
            'experiments.lora_kv_subject_v1.attention': SimpleNamespace(install_subject_lora_kv=
                lambda *args, **kwargs: owner.installs.append(('single_lora', args, kwargs))),
            'experiments.lora_kv_material_v1.attention': SimpleNamespace(install_dual_token_local_lora_kv=
                lambda *args, **kwargs: owner.installs.append(('dual_lora', args, kwargs)))}

    def model_json(self, path):
        model = {'root': str(self.root / 'model')}
        return {'model': model, 'files': {}} if Path(path).name == 'model_source.json' else {'model': model}

    def selection(self, token, method):
        folder = self.root / f'{method}-{token[1]}'
        folder.mkdir(exist_ok=True)
        weight = 'pytorch_token_local_kv_weights.bin' if method == 'fullrank' else 'pytorch_lora_kv_weights.bin'
        for name in (weight, 'token.bin'):
            (folder / name).write_text(token)
        return {'method': method, 'token': token, 'branch': 'subject' if token == '<S*>' else 'color',
            'status': 'verified', 'steps': 1000, 'training_steps': 1000, 'paired_embedding_finite': True,
            'checkpoint': str(folder), 'weight': weight,
            'hashes': {str(path): inference.sha256(path) for path in folder.iterdir()}}

    def test_real_loader_routes_both_methods_to_distinct_subject_color_slots(self):
        modules = self.modules()
        with patch.dict(sys.modules, modules), patch.object(inference.evaluate, 'read_json', side_effect=self.model_json):
            for method in prompts.METHODS:
                subject, color = self.selection('<S*>', method), self.selection('<C*>', method)
                single = inference.evaluate.load_pipeline([color], 'cpu')
                self.assertEqual(single.tokens, ['<C*>.bin'])
                joint = inference.evaluate.load_pipeline([subject, color], 'cpu')
                self.assertEqual(joint.tokens, ['<S*>.bin', '<C*>.bin'])
                masks = inference.evaluate.attention_masks(joint, prompts.JOINT_PROMPTS[0], ['<S*>', '<C*>'], 6.0)['modifier_token_mask']
                self.assertEqual(set(masks), {'subject', 'material'})
                self.assertEqual(int(masks['subject'][0].sum()), 0)
                self.assertEqual(int(masks['material'][0].sum()), 0)
                self.assertEqual(int(masks['subject'][1].sum()), 1)
                self.assertEqual(int(masks['material'][1].sum()), 1)
                self.assertFalse(np.any(masks['subject'] & masks['material']))
                with self.assertRaises(ValueError):
                    inference.evaluate.attention_masks(joint, '<S*> <C*> <C*>', ['<S*>', '<C*>'], 6.0)
            self.assertTrue(any(row[0] == 'dual_fullrank' for row in self.installs))
            self.assertTrue(any(row[0] == 'dual_lora' for row in self.installs))
            with self.assertRaises(ValueError):
                inference.evaluate.load_pipeline([self.selection('<S*>', 'fullrank'), self.selection('<C*>', 'lora_r64')], 'cpu')

    def dataset(self):
        dataset = self.root / 'dataset'
        dataset.mkdir()
        rows = []
        for number in range(9):
            image = dataset / f'actual{number}.png'
            Image.new('RGB', (512, 512), 'orange').save(image)
            rows.append({'id': str(number), 'image': str(image), 'image_sha256': inference.sha256(image),
                         'caption': 'a photo of sphere shape in <C*> color'})
        (dataset / 'training_assets_manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return dataset

    def test_dry_run_needs_no_training_or_torch_and_displays_actual_nine(self):
        dataset, output = self.dataset(), self.root / 'dryrun'
        with patch.object(inference.data, 'verify_dataset'), patch.object(inference, 'git_info', return_value={'commit': 'test'}):
            state = inference.run(dataset, output, 'fullrank', 400)
        self.assertEqual(state['status'], 'dry_run')
        self.assertEqual(state['expected_images'], 403)
        self.assertEqual(len(list((output / 'training_references').glob('*.png'))), 9)
        self.assertEqual(len(list(output.glob('transfer*.html'))), 10)
        self.assertEqual(len(list(output.glob('subject_color*.html'))), 3)
        self.assertFalse((output / 'generation_status.jsonl').exists())

    def test_prepared_subject_rejects_changed_weights_and_wrong_token(self):
        runs = [self.selection('<S*>', method) for method in prompts.METHODS]
        locks = {path: digest for row in runs for path, digest in row['hashes'].items()}
        index = self.root / 'subject_index.json'
        record = {'status': 'verified', 'subject_step': 1000, 'training_git': inference.subject_source.TRAINING_GIT,
                  'runs': runs, 'source_locks': locks}
        inference.write(index, record)
        self.assertEqual(inference.subject_selection(index, 'fullrank')[0], runs[0])
        Path(next(iter(runs[0]['hashes']))).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            inference.subject_selection(index, 'fullrank')
        record['runs'][1]['token'] = '<C*>'
        inference.write(index, record)
        with self.assertRaisesRegex(ValueError, 'Subject adapter'):
            inference.subject_selection(index, 'lora_r64')

    def execution_sources(self, dataset):
        runs = [self.selection('<S*>', method) for method in prompts.METHODS]
        subject_index = self.root / 'subject_index.json'
        inference.write(subject_index, {'status': 'verified', 'subject_step': 1000,
            'training_git': inference.subject_source.TRAINING_GIT, 'runs': runs,
            'source_locks': {path: digest for row in runs for path, digest in row['hashes'].items()}})
        color = {**self.selection('<C*>', 'fullrank'), 'run': str(self.root / 'color_run')}
        training_index = self.root / 'color_index.json'
        inference.write(training_index, {'git': {'commit': 'test'}, 'dataset': str(dataset), 'runs': [color]})
        return color, training_index, subject_index

    def test_actual_run_loads_only_color_then_both_adapters_and_writes403_rows(self):
        dataset, modules = self.dataset(), self.modules()
        color, color_index, subject_index = self.execution_sources(dataset)
        output = self.root / 'executed'
        with patch.dict(sys.modules, modules), \
             patch.object(inference.data, 'verify_dataset'), \
             patch.object(inference, 'git_info', return_value={'commit': 'test'}), \
             patch.object(inference.training, 'audit_run', return_value=color) as audit, \
             patch.object(inference.training, 'export_step', return_value={**color, 'steps': 400}) as export, \
             patch.object(inference.evaluate, 'read_json', side_effect=self.model_json):
            state = inference.run(dataset, output, 'fullrank', 400, color_index, subject_index, True, 'cpu')
        audit.assert_called_once_with('fullrank', Path(color['run']), dataset)
        self.assertEqual(export.call_args.args[1], 400)
        self.assertEqual(state['status'], 'succeeded')
        self.assertEqual(state['generated_images'], 403)
        self.assertEqual(len(self.calls), 403)
        self.assertTrue(all(tokens == ('<C*>.bin',) for tokens, _, _ in self.calls[:310]))
        self.assertTrue(all(tokens == ('<S*>.bin', '<C*>.bin') for tokens, _, _ in self.calls[310:]))
        self.assertTrue(all(args['guidance_scale'] == 6.0 for _, _, args in self.calls))
        self.assertEqual({args['generator'] for _, _, args in self.calls}, set(range(30, 61)))
        ledger = [json.loads(line) for line in (output / 'generation_status.jsonl').read_text().splitlines()]
        self.assertEqual(len(ledger), 403)
        self.assertTrue(all(row['status'] == 'ok' and row['image_sha256'] for row in ledger))

    def test_color_index_cannot_relabel_subject_training_as_color(self):
        dataset = self.dataset()
        color, index, _ = self.execution_sources(dataset)
        record = json.loads(index.read_text())
        record['runs'][0]['token'] = '<S*>'
        inference.write(index, record)
        with patch.object(inference, 'git_info', return_value={'commit': 'test'}), \
             patch.object(inference.training, 'audit_run') as audit, \
             self.assertRaisesRegex(ValueError, 'actual Color'):
            inference.color_selection(index, dataset, 'fullrank')
        audit.assert_not_called()


if __name__ == '__main__':
    unittest.main()
