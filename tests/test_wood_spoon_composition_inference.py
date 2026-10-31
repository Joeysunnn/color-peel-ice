"""Exercise real dual-adapter dispatch and immutable wood-spoon composition inputs."""

import copy
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

from experiments.wood_spoon_composition_v1 import inference


class Array(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def to(self, device):
        return self


class Tokenizer:
    model_max_length = 77

    def convert_tokens_to_ids(self, token):
        return {'<S*>': 1, '<M*>': 2, '<T*>': 3}[token]

    def __call__(self, prompt, **kwargs):
        ids = [self.convert_tokens_to_ids(word) if word in inference.TOKENS.values() else 4 for word in prompt.split()]
        return SimpleNamespace(input_ids=Array([ids + [0] * (77 - len(ids))]))


class WoodSpoonCompositionInferenceTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix='spoon-composition-', dir=os.environ.get('TMP'))
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name).resolve()
        self.git = {'branch': 'codex/composition-test', 'commit': 'inference'}
        self.calls, self.loads, self.installs, self.embeddings = [], [], [], []
        self.audit_path = self.root / 'prepared_audit.json'
        self.audit = {'sources': {}, 'source_locks': {}, 'training_gits': {'subject': {'commit': 'bb84ce1'}},
                      'references': {}, 'inference_git': self.git}
        for branch in ('subject', 'material', 'texture'):
            self.audit['sources'][branch] = {}
            for method in inference.METHODS:
                self.audit['sources'][branch][method] = self.selection(method, branch)
            image = self.root / f'{branch}_actual.png'
            Image.new('RGB', (32, 32), 'blue').save(image)
            digest = inference.sha256(image)
            self.audit['source_locks'][str(image)] = digest
            token = inference.TOKENS[branch]
            self.audit['references'][branch] = [
                {'image': str(image), 'image_sha256': digest, 'caption': f'a photo of {token} training example {i}'}
                for i in range(inference.COUNTS[branch])]
        inference.write(self.audit_path, self.audit)
        environment = patch.dict(os.environ, {'COLORPEEL_OUTPUT_ROOT': str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)
        revision = patch.object(inference, 'git_info', return_value=self.git)
        revision.start()
        self.addCleanup(revision.stop)
        # Keep the production Linux token paths in receipts, using safe Windows fixture filenames.
        digest = patch.object(inference, 'sha256', side_effect=lambda path: self.real_sha(self.safe_path(path)))
        self.real_sha = inference.sha256
        digest.start()
        self.addCleanup(digest.stop)

    @staticmethod
    def safe_path(path):
        path = Path(path)
        return path.with_name(path.name.replace('<S*>', 'subject').replace('<M*>', 'material').replace('<T*>', 'texture'))

    def selection(self, method, branch):
        token = inference.TOKENS[branch]
        run = self.root / 'sources' / method / branch / 'training'
        checkpoint = self.root / 'sources' / method / branch / 'step_export'
        run.mkdir(parents=True)
        checkpoint.mkdir()
        step = 600 if branch == 'subject' else 1000
        initializer = {'subject': 'spoon', 'material': 'material', 'texture': 'texture'}[branch]
        config = {'stage': 'train', 'run': {'study': 'fixture', 'variant': branch, 'seed': 42},
                  'args': {'initializer_token': initializer, 'modifier_token': token, 'max_train_steps': step}}
        inference.write(run / 'config.yaml', config)
        mode = 'token_local_kv' if method == 'fullrank' else (
            'lora_material_kv' if branch == 'material' else 'lora_subject_kv')
        adaptation = {'modifier_tokens': [token], 'adaptation_mode': mode}
        if method == 'lora_r64':
            adaptation.update(rank=64, alpha=64, mode='token_local_kv')
        inference.write(checkpoint / 'adaptation_config.json', adaptation)
        weight = 'pytorch_token_local_kv_weights.bin' if method == 'fullrank' else 'pytorch_lora_kv_weights.bin'
        (checkpoint / weight).write_text(method + branch + ' actual adapter')
        self.safe_path(checkpoint / (token + '.bin')).write_text(token + ' actual embedding')
        paths = [run / 'config.yaml', checkpoint / 'adaptation_config.json', checkpoint / weight,
                 checkpoint / (token + '.bin')]
        hashes = {str(path): inference.sha256(self.safe_path(path)) for path in paths}
        self.audit['source_locks'].update(hashes)
        return {'method': method, 'branch': branch, 'token': token, 'status': 'verified', 'step': step,
                'steps': step, 'training_steps': step, 'weight': weight, 'paired_embedding_finite': True,
                'concept_id': 'wood_spoon' if branch == 'subject' else 'wood_spoon_' + branch,
                'run': str(run), 'checkpoint': str(checkpoint), 'hashes': hashes}

    def modules(self):
        owner = self

        class Scheduler:
            pass

        class UNet:
            device = 'cpu'

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def load_attn_procs(self, *args, **kwargs):
                owner.installs.append(('subject_fullrank', args, kwargs))

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
                owner.embeddings.append((checkpoint, weight_name))

            def __call__(self, prompt, **kwargs):
                owner.calls.append((tuple(self.tokens), prompt, kwargs))
                return SimpleNamespace(images=[Image.new('RGB', (512, 512), 'orange')],
                                       nsfw_content_detected=[kwargs['generator'] == 30])

        def load(path, **kwargs):
            owner.loads.append(str(path))
            return {'source': str(path)}

        torch = SimpleNamespace(float16='fp16', load=load,
            Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: seed),
            zeros_like=lambda item: Array(np.zeros_like(item)), cat=lambda items: Array(np.concatenate(items)),
            cuda=SimpleNamespace(empty_cache=lambda: None))
        return {'torch': torch, 'diffusers': SimpleNamespace(DiffusionPipeline=Pipe, PNDMScheduler=Scheduler),
            'custom_attention.unet_2d_condition_custom': SimpleNamespace(UNet2DConditionModel=UNet),
            'src.methods.colorpeel_ice.dual_token_local_kv': SimpleNamespace(install_dual_token_local_kv=
                lambda *args, **kwargs: owner.installs.append(('dual_fullrank', args, kwargs))),
            'experiments.lora_kv_subject_v1.attention': SimpleNamespace(install_subject_lora_kv=
                lambda *args, **kwargs: owner.installs.append(('single_lora', args, kwargs))),
            'experiments.lora_kv_material_v1.attention': SimpleNamespace(install_dual_token_local_lora_kv=
                lambda *args, **kwargs: owner.installs.append(('dual_lora', args, kwargs)))}

    def model_json(self, path):
        model = {'root': str(self.root / 'base_model')}
        return {'model': model, 'files': {}} if Path(path).name == 'model_source.json' else {'model': model}

    def test_exact124_image_scope_and_two_literal_prompts(self):
        rows = [row for method in inference.METHODS for branch in inference.BRANCHES
                for row in inference.prompt_rows(method, branch)]
        self.assertEqual(len(rows), 124)
        self.assertEqual(len({row['id'] for row in rows}), 124)
        self.assertEqual({row['prompt'] for row in rows}, {
            'a photo of <S*> spoon made of <M*>', 'a photo of <S*> spoon with <T*> texture'})
        self.assertEqual(inference.SEEDS, tuple(range(30, 61)))
        self.assertEqual(inference.SAMPLING['guidance_scale'], 3.5)
        for method, branch in (('original', 'material'), ('fullrank', 'subject')):
            with self.assertRaises(ValueError):
                inference.prompt_rows(method, branch)

    def test_real_loader_and_masks_route_both_methods_and_both_attribute_tokens(self):
        with patch.dict(sys.modules, self.modules()), \
             patch.object(inference.evaluate, 'read_json', side_effect=self.model_json):
            for method in inference.METHODS:
                for branch in inference.BRANCHES:
                    self.calls.clear()
                    self.loads.clear()
                    self.installs.clear()
                    output = self.root / 'jobs' / method / branch
                    proof = inference.run(self.audit_path, output, method, branch, 'cpu', True)
                    token = inference.TOKENS[branch]
                    self.assertEqual(proof['generated_images'], 31)
                    self.assertEqual(proof['safety_filtered'], 1)
                    self.assertFalse(proof['retraining'])
                    self.assertFalse(proof['reexporting'])
                    self.assertEqual(proof['internal_adapter_slots'], {'subject': '<S*>', 'material': token})
                    self.assertEqual(len(self.calls), 31)
                    self.assertEqual([call[2]['generator'] for call in self.calls], list(range(30, 61)))
                    for tokens, prompt, kwargs in self.calls:
                        self.assertEqual(tokens, ('<S*>.bin', token + '.bin'))
                        self.assertEqual(prompt, inference.PROMPTS[branch])
                        self.assertEqual((kwargs['num_inference_steps'], kwargs['guidance_scale'], kwargs['width'], kwargs['height']), (100, 3.5, 512, 512))
                        masks = kwargs['cross_attention_kwargs']['modifier_token_mask']
                        self.assertEqual(set(masks), {'subject', 'material'})
                        self.assertEqual([int(masks[key][0].sum()) for key in masks], [0, 0])
                        self.assertEqual([int(masks[key][1].sum()) for key in masks], [1, 1])
                        self.assertFalse(np.any(masks['subject'] & masks['material']))
                    subject, attribute = [self.audit['sources'][key][method] for key in ('subject', branch)]
                    if method == 'fullrank':
                        self.assertEqual(self.installs[0][0], 'subject_fullrank')
                        self.assertEqual(self.installs[0][1], (subject['checkpoint'],))
                        self.assertEqual(self.installs[1][0], 'dual_fullrank')
                        self.assertEqual(self.installs[1][2], {'primary_label': 'subject'})
                        self.assertEqual(self.loads, [str(Path(attribute['checkpoint']) / attribute['weight'])])
                    else:
                        self.assertEqual(self.installs[0][0], 'dual_lora')
                        self.assertEqual(self.installs[0][2], {'rank': 64, 'alpha': 64})
                        self.assertEqual(self.loads, [str(Path(item['checkpoint']) / item['weight']) for item in (subject, attribute)])
                    records = [json.loads(line) for line in (output / 'generation_status.jsonl').read_text().splitlines()]
                    self.assertEqual(len(records), 31)
                    self.assertTrue(all(row['subject_adapter_loaded'] and row['attribute_adapter_loaded'] for row in records))
                    self.assertTrue(all(inference.sha256(output / row['image_path']) == row['image_sha256'] for row in records))
                    refs = json.loads((output / 'training_references.json').read_text())['references']
                    self.assertEqual({key: len(value) for key, value in refs.items()}, {'subject': 72, branch: inference.COUNTS[branch]})
                    self.assertTrue((output / f'references_{branch}.html').is_file())
                    self.assertEqual(len(list(output.glob('images/*.png'))), 31)

    def snapshot_audit(self, branch, step):
        for method in inference.METHODS:
            item = self.audit['sources'][branch][method]
            original = Path(item['checkpoint'])
            checkpoint = original.with_name(f'step{step}_export')
            checkpoint.mkdir()
            inference.write(checkpoint / 'adaptation_config.json',
                            json.loads((original / 'adaptation_config.json').read_text()))
            (checkpoint / item['weight']).write_text(f'{method} actual {branch}{step} adapter')
            token = inference.TOKENS[branch]
            self.safe_path(checkpoint / (token + '.bin')).write_text(f'{method} actual {branch}{step} embedding')
            item.update(checkpoint=str(checkpoint), step=step, steps=step)
            paths = [Path(item['run']) / 'config.yaml', checkpoint / 'adaptation_config.json',
                     checkpoint / item['weight'], checkpoint / (token + '.bin')]
            item['hashes'] = {str(path): inference.sha256(path) for path in paths}
            self.audit['source_locks'].update(item['hashes'])
        inference.write(self.audit_path, self.audit)

    def texture500_audit(self):
        self.snapshot_audit('texture', 500)

    def subject180_audit(self):
        reference = self.audit['references']['subject'][0]
        self.audit['references']['subject'] = [{**reference, 'caption': f'a photo of <S*> training example {i}'}
                                                for i in range(180)]
        for method in inference.METHODS:
            subject = self.audit['sources']['subject'][method]
            config_path = Path(subject['run']) / 'config.yaml'
            config = json.loads(config_path.read_text())
            config['args']['max_train_steps'] = 1000
            inference.write(config_path, config)
            subject['training_steps'] = 1000
            subject['hashes'][str(config_path)] = self.audit['source_locks'][str(config_path)] = inference.sha256(config_path)
        inference.write(self.audit_path, self.audit)

    def literal_texture_audit(self):
        self.texture500_audit()
        textures = self.audit['sources']['texture']
        refs = self.audit['references']['texture']
        self.audit['sources'] = {'texture': textures}
        self.audit['references'] = {'texture': refs}
        paths = {path for item in textures.values() for path in item['hashes']}
        paths.update(row['image'] for row in refs)
        self.audit['source_locks'] = {path: self.audit['source_locks'][path] for path in paths}
        self.audit['training_gits'] = {'texture': {'commit': 'original-texture'}}
        inference.write(self.audit_path, self.audit)

    def test_literal_prompt_uses_only_texture500_and_ignores_subject_arguments(self):
        for method in inference.METHODS:
            rows = inference.prompt_rows(method, 'texture', attribute_step=500, literal_subject=True)
            self.assertEqual(rows, inference.prompt_rows(method, 'texture', 500, 750, 180, True))
            self.assertEqual(len(rows), 31)
            self.assertEqual([row['seed'] for row in rows], list(range(30, 61)))
            self.assertTrue(all(row['prompt'] == 'a photo of a wood spoon with <T*> texture' for row in rows))
            self.assertTrue(all(row['subject_step'] is None and row['subject_images'] == 0 for row in rows))
            self.assertTrue(all(row['condition'] == 'literal_texture' for row in rows))
            self.assertEqual(inference.prompt_rows(method, 'texture'),
                             inference.prompt_rows(method, 'texture', literal_subject=False))
        for method, branch, step in (('fullrank', 'material', 1000), ('fullrank', 'texture', 750),
                                    ('lora_r64', 'texture', 1000), ('wrong', 'texture', 500)):
            with self.subTest(method=method, branch=branch, step=step), self.assertRaisesRegex(ValueError, 'Texture500'):
                inference.prompt_rows(method, branch, attribute_step=step, literal_subject=True)

    def test_literal_real_loader_uses_only_texture_adapter_embedding_and_single_mask_for_both_methods(self):
        self.literal_texture_audit()
        with patch.dict(sys.modules, self.modules()), \
             patch.object(inference.evaluate, 'read_json', side_effect=self.model_json):
            for method in inference.METHODS:
                self.calls.clear()
                self.loads.clear()
                self.installs.clear()
                self.embeddings.clear()
                texture = self.audit['sources']['texture'][method]
                self.assertEqual(inference.selections(self.audit, method, 'texture', 500, literal_subject=True), [texture])
                output = self.root / 'literal' / method
                proof = inference.run(self.audit_path, output, method, 'texture', 'cpu', True,
                                      attribute_step=500, literal_subject=True)
                self.assertTrue(proof['literal_subject'])
                self.assertEqual((proof['subject_step'], proof['subject_images'], proof['subject_training']), (None, 0, None))
                self.assertEqual(proof['attribute_training'], texture)
                self.assertEqual(proof['internal_adapter_slots'], {'modifier': '<T*>'})
                self.assertEqual(proof['training_images'], {'texture': 16})
                self.assertEqual(proof['generated_images'], 31)
                self.assertEqual(self.embeddings, [(texture['checkpoint'], '<T*>.bin')])
                self.assertEqual(len(self.installs), 1)
                if method == 'fullrank':
                    self.assertEqual(self.installs[0][1], (texture['checkpoint'],))
                    self.assertEqual(self.installs[0][2]['weight_name'], texture['weight'])
                    self.assertEqual(self.loads, [])
                else:
                    self.assertEqual(self.installs[0][0], 'single_lora')
                    self.assertEqual(self.installs[0][1][1:], ({'source': str(Path(texture['checkpoint']) / texture['weight'])},
                                                            'token_local_kv', 64, 64))
                    self.assertEqual(self.loads, [str(Path(texture['checkpoint']) / texture['weight'])])
                self.assertEqual([call[2]['generator'] for call in self.calls], list(range(30, 61)))
                for tokens, prompt, kwargs in self.calls:
                    self.assertEqual(tokens, ('<T*>.bin',))
                    self.assertEqual(prompt, 'a photo of a wood spoon with <T*> texture')
                    self.assertNotIn('<S*>', prompt)
                    self.assertEqual((kwargs['num_inference_steps'], kwargs['guidance_scale'], kwargs['width'], kwargs['height']), (100, 3.5, 512, 512))
                    mask = kwargs['cross_attention_kwargs']['modifier_token_mask']
                    self.assertNotIsInstance(mask, dict)
                    self.assertEqual(mask.dtype, np.bool_)
                    self.assertEqual(mask.shape, (2, 77))
                    self.assertEqual((int(mask[0].sum()), int(mask[1].sum())), (0, 1))
                records = [json.loads(line) for line in (output / 'generation_status.jsonl').read_text().splitlines()]
                self.assertTrue(all(row['subject_adapter_loaded'] is False and row['attribute_adapter_loaded'] is True for row in records))
                refs = json.loads((output / 'training_references.json').read_text())['references']
                self.assertEqual(set(refs), {'texture'})
                self.assertEqual(len(refs['texture']), 16)
                self.assertFalse((output / 'training_references' / 'subject').exists())
                self.assertFalse((output / 'references_subject.html').exists())
                page = (output / 'index.html').read_text(encoding='utf8')
                self.assertIn('literal wood spoon + T500', page)
                self.assertNotIn('Subject training images: 0', page)
                self.assertNotIn('SNone', page)

    def test_literal_missing_texture_or_wrong_snapshot_fails_without_subject_fallback(self):
        self.literal_texture_audit()
        self.audit['sources']['texture']['fullrank']['steps'] = 1000
        inference.write(self.audit_path, self.audit)
        with patch.object(inference.evaluate, 'load_pipeline') as load:
            with self.assertRaisesRegex(ValueError, 'audited Texture500'):
                inference.run(self.audit_path, self.root / 'wrong-literal-step', 'fullrank', 'texture', 'cpu', True,
                              attribute_step=500, literal_subject=True)
            del self.audit['sources']['texture']['lora_r64']
            inference.write(self.audit_path, self.audit)
            with self.assertRaisesRegex(ValueError, 'audited Texture500'):
                inference.run(self.audit_path, self.root / 'missing-literal-texture', 'lora_r64', 'texture', 'cpu', True,
                              attribute_step=500, literal_subject=True)
            load.assert_not_called()

    def test_texture500_uses_its_actual_adapter_and_embedding_with_subject600_for_both_methods(self):
        self.texture500_audit()
        with patch.dict(sys.modules, self.modules()), \
             patch.object(inference.evaluate, 'read_json', side_effect=self.model_json):
            for method in inference.METHODS:
                self.calls.clear()
                self.loads.clear()
                self.installs.clear()
                self.embeddings.clear()
                output = self.root / 'texture500' / method
                proof = inference.run(self.audit_path, output, method, 'texture', 'cpu', True, attribute_step=500)
                subject = self.audit['sources']['subject'][method]
                texture = self.audit['sources']['texture'][method]
                self.assertEqual((proof['subject_step'], proof['attribute_step'], proof['generated_images']), (600, 500, 31))
                self.assertEqual(texture['training_steps'], 1000)
                self.assertEqual(self.embeddings, [(subject['checkpoint'], '<S*>.bin'), (texture['checkpoint'], '<T*>.bin')])
                self.assertEqual(self.loads, [str(Path(item['checkpoint']) / item['weight'])
                    for item in ((texture,) if method == 'fullrank' else (subject, texture))])
                self.assertEqual(self.installs[-1][0], 'dual_fullrank' if method == 'fullrank' else 'dual_lora')
                self.assertEqual([call[2]['generator'] for call in self.calls], list(range(30, 61)))
                for tokens, prompt, kwargs in self.calls:
                    self.assertEqual(tokens, ('<S*>.bin', '<T*>.bin'))
                    self.assertEqual(prompt, 'a photo of <S*> spoon with <T*> texture')
                    self.assertEqual((kwargs['num_inference_steps'], kwargs['guidance_scale']), (100, 3.5))
                    masks = kwargs['cross_attention_kwargs']['modifier_token_mask']
                    self.assertEqual([int(masks[key][0].sum()) for key in masks], [0, 0])
                    self.assertEqual([int(masks[key][1].sum()) for key in masks], [1, 1])
                rows = [json.loads(line) for line in (output / 'generation_manifest.jsonl').read_text().splitlines()]
                self.assertTrue(all(row['subject_step'] == 600 and row['attribute_step'] == 500 for row in rows))
                self.assertIn('S600 + T500', (output / 'index.html').read_text(encoding='utf8'))

    def test_texture1000_cannot_substitute_for_requested500_and_material_remains1000(self):
        with patch.object(inference.evaluate, 'load_pipeline') as load:
            for method in inference.METHODS:
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'attribute500'):
                    inference.run(self.audit_path, self.root / ('wrong-texture-' + method), method,
                                  'texture', 'cpu', True, attribute_step=500)
            for branch, step in (('material', 500), ('texture', 900), ('texture', 500.0), ('texture', True)):
                with self.subTest(branch=branch, step=step), self.assertRaises(ValueError):
                    inference.run(self.audit_path, self.root / 'invalid-step', 'fullrank', branch,
                                  'cpu', True, attribute_step=step)
            load.assert_not_called()

    def test_texture500_still_requires1000_training_steps_and1000_training_config(self):
        self.texture500_audit()
        texture = self.audit['sources']['texture']['fullrank']
        texture['training_steps'] = 500
        inference.write(self.audit_path, self.audit)
        with self.assertRaisesRegex(ValueError, 'attribute500'):
            inference.run(self.audit_path, self.root / 'wrong-training-total', 'fullrank', 'texture', attribute_step=500)
        texture['training_steps'] = 1000
        config_path = Path(texture['run']) / 'config.yaml'
        config = json.loads(config_path.read_text())
        config['args']['max_train_steps'] = 500
        inference.write(config_path, config)
        texture['hashes'][str(config_path)] = self.audit['source_locks'][str(config_path)] = inference.sha256(config_path)
        inference.write(self.audit_path, self.audit)
        with self.assertRaisesRegex(ValueError, 'source token, steps or initializer'):
            inference.run(self.audit_path, self.root / 'wrong-training-config', 'fullrank', 'texture', attribute_step=500)

    def test_subject200_and400_use_same_step_adapters_and_embeddings_with_texture500(self):
        self.texture500_audit()
        with patch.dict(sys.modules, self.modules()), \
             patch.object(inference.evaluate, 'read_json', side_effect=self.model_json):
            for step in (200, 400):
                self.snapshot_audit('subject', step)
                for method in inference.METHODS:
                    self.calls.clear()
                    self.loads.clear()
                    self.installs.clear()
                    self.embeddings.clear()
                    output = self.root / f'subject{step}' / method
                    proof = inference.run(self.audit_path, output, method, 'texture', 'cpu', True,
                                          attribute_step=500, subject_step=step)
                    subject = self.audit['sources']['subject'][method]
                    texture = self.audit['sources']['texture'][method]
                    self.assertEqual((proof['subject_step'], proof['attribute_step'], proof['generated_images']), (step, 500, 31))
                    self.assertEqual(subject['training_steps'], 600)
                    self.assertEqual(texture['training_steps'], 1000)
                    self.assertEqual(self.embeddings, [(subject['checkpoint'], '<S*>.bin'), (texture['checkpoint'], '<T*>.bin')])
                    self.assertEqual(self.loads, [str(Path(item['checkpoint']) / item['weight'])
                        for item in ((texture,) if method == 'fullrank' else (subject, texture))])
                    if method == 'fullrank':
                        self.assertEqual(self.installs[0][1], (subject['checkpoint'],))
                    self.assertEqual(self.installs[-1][0], 'dual_fullrank' if method == 'fullrank' else 'dual_lora')
                    self.assertEqual([call[2]['generator'] for call in self.calls], list(range(30, 61)))
                    for tokens, prompt, kwargs in self.calls:
                        self.assertEqual(tokens, ('<S*>.bin', '<T*>.bin'))
                        self.assertEqual(prompt, inference.PROMPTS['texture'])
                        self.assertEqual((kwargs['num_inference_steps'], kwargs['guidance_scale']), (100, 3.5))
                        masks = kwargs['cross_attention_kwargs']['modifier_token_mask']
                        self.assertEqual([int(masks[key][0].sum()) for key in masks], [0, 0])
                        self.assertEqual([int(masks[key][1].sum()) for key in masks], [1, 1])
                    rows = [json.loads(line) for line in (output / 'generation_manifest.jsonl').read_text().splitlines()]
                    self.assertTrue(all(row['subject_step'] == step and row['attribute_step'] == 500 for row in rows))
                    self.assertIn(f'S{step} + T500', (output / 'index.html').read_text(encoding='utf8'))

    def test_texture750_routes_paired_adapters_and_tokens_with_all_three_subject_steps(self):
        self.snapshot_audit('texture', 750)
        generated = 0
        with patch.dict(sys.modules, self.modules()), \
             patch.object(inference.evaluate, 'read_json', side_effect=self.model_json):
            for step in (200, 400, 600):
                self.snapshot_audit('subject', step)
                for method in inference.METHODS:
                    with self.subTest(step=step, method=method):
                        self.calls.clear()
                        self.loads.clear()
                        self.installs.clear()
                        self.embeddings.clear()
                        subject = self.audit['sources']['subject'][method]
                        texture = self.audit['sources']['texture'][method]
                        selected = inference.selections(self.audit, method, 'texture',
                                                        attribute_step=750, subject_step=step)
                        self.assertEqual(selected, [subject, texture])
                        self.assertEqual((selected[0]['step'], selected[0]['training_steps']), (step, 600))
                        self.assertEqual((selected[1]['step'], selected[1]['steps'], selected[1]['training_steps']), (750, 750, 1000))
                        output = self.root / 'texture750' / f'subject{step}' / method
                        proof = inference.run(self.audit_path, output, method, 'texture', 'cpu', True,
                                              attribute_step=750, subject_step=step)
                        generated += proof['generated_images']
                        self.assertEqual((proof['subject_step'], proof['attribute_step'], proof['generated_images']), (step, 750, 31))
                        self.assertEqual(proof['subject_training'], subject)
                        self.assertEqual(proof['attribute_training'], texture)
                        self.assertEqual(self.embeddings, [(subject['checkpoint'], '<S*>.bin'), (texture['checkpoint'], '<T*>.bin')])
                        self.assertEqual(self.loads, [str(Path(item['checkpoint']) / item['weight'])
                            for item in ((texture,) if method == 'fullrank' else (subject, texture))])
                        if method == 'fullrank':
                            self.assertEqual(self.installs[0][1], (subject['checkpoint'],))
                        self.assertEqual(self.installs[-1][0], 'dual_fullrank' if method == 'fullrank' else 'dual_lora')
                        self.assertEqual([call[2]['generator'] for call in self.calls], list(range(30, 61)))
                        for tokens, prompt, kwargs in self.calls:
                            self.assertEqual(tokens, ('<S*>.bin', '<T*>.bin'))
                            self.assertEqual(prompt, 'a photo of <S*> spoon with <T*> texture')
                            self.assertEqual((kwargs['num_inference_steps'], kwargs['guidance_scale']), (100, 3.5))
                            masks = kwargs['cross_attention_kwargs']['modifier_token_mask']
                            self.assertEqual([int(masks[key][0].sum()) for key in masks], [0, 0])
                            self.assertEqual([int(masks[key][1].sum()) for key in masks], [1, 1])
                        rows = [json.loads(line) for line in (output / 'generation_manifest.jsonl').read_text().splitlines()]
                        self.assertEqual(len(rows), 31)
                        self.assertTrue(all(row['subject_step'] == step and row['attribute_step'] == 750 for row in rows))
                        self.assertIn(f'S{step} + T750', (output / 'index.html').read_text(encoding='utf8'))
        self.assertEqual(generated, 186)

    def test_texture500_and1000_cannot_replace_requested750(self):
        with patch.object(inference.evaluate, 'load_pipeline') as load:
            for actual in (1000, 500):
                if actual == 500:
                    self.snapshot_audit('texture', 500)
                for method in inference.METHODS:
                    output = self.root / f'wrong-texture{actual}-{method}'
                    with self.subTest(actual=actual, method=method), self.assertRaisesRegex(ValueError, 'attribute750'):
                        inference.run(self.audit_path, output, method, 'texture', 'cpu', True, attribute_step=750)
                    self.assertFalse(output.exists())
            load.assert_not_called()

    def test_subject180_at750_and1000_routes_both_methods_with_texture500_and_copies180_references(self):
        self.subject180_audit()
        self.texture500_audit()
        with patch.dict(sys.modules, self.modules()), \
             patch.object(inference.evaluate, 'read_json', side_effect=self.model_json):
            for step in (750, 1000):
                self.snapshot_audit('subject', step)
                for method in inference.METHODS:
                    with self.subTest(step=step, method=method):
                        self.calls.clear()
                        self.loads.clear()
                        self.installs.clear()
                        self.embeddings.clear()
                        subject = self.audit['sources']['subject'][method]
                        texture = self.audit['sources']['texture'][method]
                        self.assertEqual(inference.selections(self.audit, method, 'texture', 500, step, 180), [subject, texture])
                        output = self.root / 'subject180' / f'step{step}' / method
                        proof = inference.run(self.audit_path, output, method, 'texture', 'cpu', True,
                                              attribute_step=500, subject_step=step, subject_images=180)
                        self.assertEqual((proof['subject_images'], proof['subject_step'], proof['attribute_step']), (180, step, 500))
                        self.assertEqual(proof['subject_training']['training_steps'], 1000)
                        self.assertEqual(proof['training_images'], {'subject': 180, 'texture': 16})
                        self.assertEqual(proof['generated_images'], 31)
                        self.assertEqual(self.embeddings, [(subject['checkpoint'], '<S*>.bin'), (texture['checkpoint'], '<T*>.bin')])
                        self.assertEqual(self.loads, [str(Path(item['checkpoint']) / item['weight'])
                            for item in ((texture,) if method == 'fullrank' else (subject, texture))])
                        if method == 'fullrank':
                            self.assertEqual(self.installs[0][1], (subject['checkpoint'],))
                        self.assertEqual(self.installs[-1][0], 'dual_fullrank' if method == 'fullrank' else 'dual_lora')
                        self.assertEqual([call[2]['generator'] for call in self.calls], list(range(30, 61)))
                        for tokens, prompt, kwargs in self.calls:
                            self.assertEqual(tokens, ('<S*>.bin', '<T*>.bin'))
                            self.assertEqual(prompt, inference.PROMPTS['texture'])
                            self.assertEqual((kwargs['num_inference_steps'], kwargs['guidance_scale']), (100, 3.5))
                            masks = kwargs['cross_attention_kwargs']['modifier_token_mask']
                            self.assertEqual([int(masks[key][0].sum()) for key in masks], [0, 0])
                            self.assertEqual([int(masks[key][1].sum()) for key in masks], [1, 1])
                        rows = [json.loads(line) for line in (output / 'generation_manifest.jsonl').read_text().splitlines()]
                        self.assertTrue(all(row['subject_images'] == 180 and row['subject_step'] == step
                                            and row['attribute_step'] == 500 for row in rows))
                        refs = json.loads((output / 'training_references.json').read_text())['references']
                        self.assertEqual(refs['subject'], [{**row, 'review_image': f'training_references/subject/train_{i:03d}.png'}
                                                          for i, row in enumerate(self.audit['references']['subject'])])
                        self.assertEqual(len(list((output / 'training_references' / 'subject').glob('*.png'))), 180)
                        page = (output / 'index.html').read_text(encoding='utf8')
                        self.assertIn(f'S{step} + T500', page)
                        self.assertIn('Subject training images: 180', page)

    def test_subject_image_count_and_step_recipe_cannot_be_mixed(self):
        for images, step in ((72, 750), (72, 1000), (180, 200), (180, 400), (180, 600), (35, 750), (180.0, 750), (True, 750)):
            with self.subTest(images=images, step=step), self.assertRaises(ValueError):
                inference.prompt_rows('fullrank', 'texture', attribute_step=500, subject_step=step, subject_images=images)
        self.subject180_audit()
        with self.assertRaisesRegex(ValueError, 'every actual selected training RGB'):
            inference.references(self.audit, 'texture', self.root / 'wrong-count72', subject_images=72)
        self.texture500_audit()
        self.snapshot_audit('subject', 750)
        self.audit['references']['subject'] = self.audit['references']['subject'][:72]
        inference.write(self.audit_path, self.audit)
        with patch.object(inference.evaluate, 'load_pipeline') as load:
            with self.assertRaisesRegex(ValueError, 'every actual selected training RGB'):
                inference.run(self.audit_path, self.root / 'missing108-references', 'fullrank', 'texture', 'cpu', True,
                              attribute_step=500, subject_step=750, subject_images=180)
            load.assert_not_called()

    def test_subject180_requires1000_training_updates_and_config(self):
        self.subject180_audit()
        self.texture500_audit()
        self.snapshot_audit('subject', 750)
        subject = self.audit['sources']['subject']['fullrank']
        subject['training_steps'] = 600
        inference.write(self.audit_path, self.audit)
        with self.assertRaisesRegex(ValueError, 'Subject750'):
            inference.run(self.audit_path, self.root / 'wrong180-total', 'fullrank', 'texture',
                          attribute_step=500, subject_step=750, subject_images=180)
        subject['training_steps'] = 1000
        config_path = Path(subject['run']) / 'config.yaml'
        config = json.loads(config_path.read_text())
        config['args']['max_train_steps'] = 600
        inference.write(config_path, config)
        subject['hashes'][str(config_path)] = self.audit['source_locks'][str(config_path)] = inference.sha256(config_path)
        inference.write(self.audit_path, self.audit)
        with self.assertRaisesRegex(ValueError, 'source token, steps or initializer'):
            inference.run(self.audit_path, self.root / 'wrong180-config', 'fullrank', 'texture',
                          attribute_step=500, subject_step=750, subject_images=180)

    def test_subject600_cannot_replace_requested200_or400_and_invalid_steps_are_rejected(self):
        self.texture500_audit()
        with patch.object(inference.evaluate, 'load_pipeline') as load:
            for step in (200, 400):
                for method in inference.METHODS:
                    with self.subTest(step=step, method=method), self.assertRaisesRegex(ValueError, f'Subject{step}'):
                        inference.run(self.audit_path, self.root / f'wrong-subject{step}-{method}', method,
                                      'texture', 'cpu', True, attribute_step=500, subject_step=step)
            for step in (500, 200.0, True):
                with self.subTest(step=step), self.assertRaises(ValueError):
                    inference.prompt_rows('fullrank', 'texture', attribute_step=500, subject_step=step)
            load.assert_not_called()

    def test_subject200_still_requires600_training_steps_and600_training_config(self):
        self.texture500_audit()
        self.snapshot_audit('subject', 200)
        subject = self.audit['sources']['subject']['fullrank']
        subject['training_steps'] = 200
        inference.write(self.audit_path, self.audit)
        with self.assertRaisesRegex(ValueError, 'Subject200'):
            inference.run(self.audit_path, self.root / 'wrong-subject-total', 'fullrank', 'texture',
                          attribute_step=500, subject_step=200)
        subject['training_steps'] = 600
        config_path = Path(subject['run']) / 'config.yaml'
        config = json.loads(config_path.read_text())
        config['args']['max_train_steps'] = 200
        inference.write(config_path, config)
        subject['hashes'][str(config_path)] = self.audit['source_locks'][str(config_path)] = inference.sha256(config_path)
        inference.write(self.audit_path, self.audit)
        with self.assertRaisesRegex(ValueError, 'source token, steps or initializer'):
            inference.run(self.audit_path, self.root / 'wrong-subject-config', 'fullrank', 'texture',
                          attribute_step=500, subject_step=200)

    def test_wrong_steps_branch_method_revision_and_missing_locks_fail_before_loading(self):
        cases = ('subject_step', 'attribute_step', 'cross_method', 'wrong_attribute', 'wrong_subject', 'git', 'missing_lock')
        for case in cases:
            audit = copy.deepcopy(self.audit)
            subject = audit['sources']['subject']['fullrank']
            attribute = audit['sources']['material']['fullrank']
            if case == 'subject_step':
                subject['steps'] = 1000
            elif case == 'attribute_step':
                attribute['training_steps'] = 600
            elif case == 'cross_method':
                attribute['method'] = 'lora_r64'
            elif case == 'wrong_attribute':
                attribute['token'] = '<T*>'
            elif case == 'wrong_subject':
                subject['concept_id'] = 'metal_spoon'
            elif case == 'git':
                audit['inference_git'] = {'commit': 'other'}
            else:
                audit['source_locks'].pop(next(iter(subject['hashes'])))
            inference.write(self.audit_path, audit)
            output = self.root / ('rejected-' + case)
            with self.subTest(case=case), patch.object(inference.evaluate, 'load_pipeline') as load:
                with self.assertRaises(ValueError):
                    inference.run(self.audit_path, output, 'fullrank', 'material', 'cpu', True)
                load.assert_not_called()
                self.assertFalse(output.exists())

    def test_changed_source_and_wrong_material_initializer_are_rejected(self):
        material = self.audit['sources']['material']['lora_r64']
        config_path = Path(material['run']) / 'config.yaml'
        config = json.loads(config_path.read_text())
        config['args']['initializer_token'] = 'wood'
        inference.write(config_path, config)
        with patch.object(inference.evaluate, 'load_pipeline') as load:
            with self.assertRaisesRegex(ValueError, 'source locks'):
                inference.run(self.audit_path, self.root / 'changed', 'lora_r64', 'material', 'cpu', True)
            digest = inference.sha256(config_path)
            self.audit['source_locks'][str(config_path)] = material['hashes'][str(config_path)] = digest
            inference.write(self.audit_path, self.audit)
            with self.assertRaisesRegex(ValueError, 'initializer'):
                inference.run(self.audit_path, self.root / 'wrong-initializer', 'lora_r64', 'material', 'cpu', True)
            load.assert_not_called()

    def test_dry_run_displays_all_references_without_loading_pipeline(self):
        output = self.root / 'dryrun'
        with patch.object(inference.evaluate, 'load_pipeline') as load:
            proof = inference.run(self.audit_path, output, 'fullrank', 'texture')
        self.assertEqual(proof['status'], 'dry_run')
        self.assertEqual(proof['expected_images'], 31)
        self.assertEqual(proof['training_images'], {'subject': 72, 'texture': 16})
        self.assertEqual(len((output / 'generation_manifest.jsonl').read_text().splitlines()), 31)
        self.assertFalse((output / 'generation_status.jsonl').exists())
        load.assert_not_called()

    def test_fresh_output_cannot_overlap_any_source(self):
        material = self.audit['sources']['material']['fullrank']
        checkpoint = Path(material['checkpoint'])
        for output in (checkpoint / 'new_result', checkpoint.parent.parent, self.root / 'sources' / 'lora_r64' / 'texture' / 'training' / 'new_result'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                inference.run(self.audit_path, output, 'fullrank', 'material')
        existing = self.root / 'existing'
        existing.mkdir()
        with self.assertRaises(ValueError):
            inference.run(self.audit_path, existing, 'fullrank', 'material')

    def test_generation_failure_is_retained_without_retry(self):
        output = self.root / 'failed'
        with patch.dict(sys.modules, self.modules()), \
             patch.object(inference.evaluate, 'load_pipeline', side_effect=RuntimeError('CUDA failed')) as load:
            with self.assertRaisesRegex(RuntimeError, 'CUDA failed'):
                inference.run(self.audit_path, output, 'fullrank', 'texture', 'cpu', True)
        self.assertEqual(load.call_count, 1)
        proof = json.loads((output / 'provenance.json').read_text())
        self.assertEqual((proof['status'], proof['generated_images']), ('failed', 0))
        self.assertTrue((output / 'index.html').is_file())


if __name__ == '__main__':
    unittest.main()
