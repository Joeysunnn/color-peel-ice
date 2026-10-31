"""CPU integration for SD2.1 paired1024 exports, independent probes and stock controls."""

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

from experiments.sd21_wood_branches_v1 import contract, inference, prepare_model, snapshots


class Tensor:
    def __init__(self, value, shape=None):
        self.value = np.asarray(value)
        self.shape = tuple(shape) if shape is not None else self.value.shape
        self.ndim = len(self.shape)

    def __getitem__(self, item):
        return Tensor(self.value[item])

    def clone(self):
        return Tensor(self.value.copy(), self.shape)

    def contiguous(self):
        return self


class Array(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def to(self, device):
        return self


class SD21WoodInferenceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='sd21-inference-', dir=os.environ.get('TMP'))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / 'sd21'
        self.model.mkdir()
        (self.model / 'weights.bin').write_bytes(b'frozen SD2.1 weights')
        self.real_sha = snapshots.sha256
        self.receipt = {'model': {'root': str(self.model), 'repo': 'sd21', 'revision': 'frozen'},
                        'files': {'weights.bin': self.real_sha(self.model / 'weights.bin')}}
        (self.model / 'model_source.json').write_text(json.dumps(self.receipt))
        self.git = {'branch': 'codex/sd21-test', 'commit': 'cpu'}
        self.registry, self.calls, self.loads, self.installs, self.generators = {}, [], [], [], []
        self.pipeline_defect = None
        self.counter = 0
        self.data = {}
        for branch, count in (('subject', 72), ('material', 8), ('texture', 16)):
            folder = self.root / branch
            folder.mkdir()
            image, mask = folder / 'actual.png', folder / 'mask.png'
            Image.new('RGB', (32, 32), 'blue').save(image)
            Image.new('L', (32, 32), 255).save(mask)
            refs = [{'id': f'{branch}{i}', 'image': str(image), 'mask': str(mask),
                     'image_sha256': self.real_sha(image), 'mask_sha256': self.real_sha(mask),
                     'caption': f'a photo of {contract.TOKENS[branch]} example{i}'} for i in range(count)]
            self.data[branch] = {'rows': count, 'references': refs,
                                'source_locks': {str(path): self.real_sha(path) for path in (image, mask)}}
        for item in (patch.dict(os.environ, {'COLORPEEL_OUTPUT_ROOT': str(self.root)}),
                     patch.object(snapshots, 'MODEL_DIR', self.model), patch.object(inference, 'MODEL_DIR', self.model),
                     patch.object(snapshots, 'git_info', return_value=self.git),
                     patch.object(inference, 'git_info', return_value=self.git),
                     patch.object(contract, 'build_config', side_effect=self.config),
                     patch.object(contract, 'verify_dataset', side_effect=lambda branch: self.data[branch]),
                     patch.object(prepare_model, 'verify_model', return_value=self.receipt),
                     patch.object(snapshots, 'sha256', side_effect=self.sha),
                     patch.object(snapshots.shared, 'sha256', side_effect=self.sha),
                     patch.object(inference, 'sha256', side_effect=self.sha)):
            item.start()
            self.addCleanup(item.stop)
        self.torch = SimpleNamespace(Tensor=Tensor, float16='float16', load=self.load, save=self.save,
            isfinite=lambda tensor: np.isfinite(tensor.value),
            equal=lambda left, right: left.shape == right.shape and np.array_equal(left.value, right.value),
            zeros_like=lambda item: Array(np.zeros_like(item)), cat=lambda items: Array(np.concatenate(items)),
            cuda=SimpleNamespace(empty_cache=lambda: None), Generator=self.generator)

    @staticmethod
    def safe(path):
        path = Path(path)
        return path.with_name(path.name.replace('<S*>', 'subject_token').replace('<M*>', 'material_token').replace('<T*>', 'texture_token'))

    def sha(self, path):
        return self.real_sha(self.safe(path))

    def save(self, value, path):
        self.registry[str(path)] = value
        def encode(item):
            if isinstance(item, Tensor):
                return {'shape': item.shape, 'value': item.value.tolist()}
            raise TypeError(type(item).__name__)
        self.safe(path).write_text(json.dumps(value, default=encode), encoding='utf8')

    def load(self, path, **kwargs):
        return self.registry[str(path)]

    def generator(self, device):
        value = SimpleNamespace(device=device, seed=None)
        def manual_seed(seed):
            value.seed = seed
            return value
        value.manual_seed = manual_seed
        self.generators.append(value)
        return value

    def config(self, branch, method, model_dir=None, smoke=False):
        return {'stage': 'train', 'run': {'study': 'sd21', 'variant': f'{branch}_{method}_{smoke}', 'seed': 42},
                'args': {'pretrained_model_name_or_path': str(model_dir or self.model),
                         'max_train_steps': 2 if smoke else 1000, 'checkpoint_steps': [2] if smoke else list(contract.STEPS),
                         'modifier_token': contract.TOKENS[branch], 'initializer_token': contract.INITIALIZERS[branch]}}

    def training_run(self, branch='subject', method='fullrank', smoke=False):
        self.counter += 1
        run = self.root / f'training{self.counter}'
        checkpoint = run / 'checkpoints'
        checkpoint.mkdir(parents=True)
        config = self.config(branch, method, smoke=smoke)
        (run / 'config.yaml').write_text(json.dumps(config))
        (run / 'manifest.json').write_text(json.dumps({'status': 'succeeded', 'returncode': 0, 'stage': 'train',
                                                      'run': config['run'], 'git': self.git}))
        maximum, steps = config['args']['max_train_steps'], config['args']['checkpoint_steps']
        token = contract.TOKENS[branch]
        mode = 'token_local_kv' if method == 'fullrank' else ('lora_material_kv' if branch == 'material' else 'lora_subject_kv')
        adaptation = {'modifier_tokens': [token], 'adaptation_mode': mode}
        if method == 'lora_r64':
            adaptation.update(mode='token_local_kv', rank=64, alpha=64)
        (checkpoint / 'adaptation_config.json').write_text(json.dumps(adaptation))
        metrics = [{'step': step, 'losses_finite': {'all': True}, 'present_modifier_tokens': [token],
                    'reconstruction_loss': 1.0, 'caa_loss': 0.0, 'caa_weighted_loss': 0.0, 'total_loss': 1.0,
                    'learning_rate': 1e-5} for step in range(1, maximum + 1)]
        (checkpoint / 'training_metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in metrics))
        audit = {'observed_optimization_steps': maximum, 'non_modifier_embedding_drift': {'changed_rows': 0},
                 'modifier_tokens': [{'token': token, 'token_id': 1, 'exposure_steps': maximum,
                    'nonzero_gradient_steps': maximum, 'initial_final_l2_delta': 1.0}]}
        (checkpoint / 'embedding_update_audit.json').write_text(json.dumps(audit))
        shapes, empty = snapshots.adapter_schema(method)
        weight = 'pytorch_token_local_kv_weights.bin' if method == 'fullrank' else 'pytorch_lora_kv_weights.bin'
        self.save({**{key: Tensor(1.0, shape) for key, shape in shapes.items()}, **empty}, checkpoint / weight)
        self.save({token: Tensor(np.ones(1024))}, checkpoint / (token + '.bin'))
        for step in steps:
            directory = checkpoint / f'checkpoint-{step}'
            directory.mkdir()
            self.save({key: Tensor(step / maximum, shape) for key, shape in shapes.items()}, directory / 'pytorch_model.bin')
            self.save({snapshots.shared.EMBEDDING_KEY: Tensor(np.stack([np.zeros(1024), np.full(1024, step / maximum)]))},
                      directory / 'pytorch_model_1.bin')
            for name in snapshots.shared.STATE_FILES[2:]:
                (directory / name).write_bytes(b'complete Accelerator state')
        return run

    def modules(self):
        owner = self
        class Processor:
            __module__ = 'diffusers.models.attention_processor'
        class UNet:
            __module__ = 'custom_attention.unet_2d_condition_custom'
            dtype, device = 'float16', 'cpu'
            def __init__(self):
                self.config = SimpleNamespace(cross_attention_dim=768 if owner.pipeline_defect == 'width' else 1024,
                    use_linear_projection=owner.pipeline_defect != 'linear')
                self.attn_processors = {'block.processor': Processor()}
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()
            def load_attn_procs(self, *args, **kwargs):
                owner.installs.append(('fullrank', args, kwargs))
        class StockUNet(UNet):
            __module__ = 'diffusers.models.unet_2d_condition'
        class Scheduler:
            config = {'prediction_type': 'epsilon'}
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                owner.assertEqual(args, (str(owner.model),))
                owner.assertEqual(kwargs, {'subfolder': 'scheduler', 'local_files_only': True})
                return cls()
        class Tokenizer:
            model_max_length = 77
            def __init__(self):
                self.vocab = {}
            def get_vocab(self):
                return self.vocab
            def convert_tokens_to_ids(self, token):
                return {'<S*>': 10, '<M*>': 11, '<T*>': 12}[token]
            def __call__(self, prompt, **kwargs):
                ids = [self.convert_tokens_to_ids(word) if word in contract.TOKENS.values() else 1 for word in prompt.split()]
                return SimpleNamespace(input_ids=Array([ids + [0] * (77 - len(ids))]))
        class Pipe:
            __module__ = 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion'
            def __init__(self, kwargs):
                self.unet, self.scheduler = kwargs.get('unet', StockUNet()), kwargs['scheduler']
                self.safety_checker, self.tokenizer, self.tokens = object(), Tokenizer(), []
                self.text_encoder = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(shape=(100, 1024))))
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                owner.loads.append((args, kwargs))
                return cls(kwargs)
            def to(self, device):
                self.unet.device = device
                return self
            def load_textual_inversion(self, checkpoint, weight_name):
                token = weight_name[:-4]
                owner.assertEqual(owner.registry[str(Path(checkpoint) / weight_name)][token].shape, (1024,))
                self.tokens.append(token)
                self.tokenizer.vocab[token] = self.tokenizer.convert_tokens_to_ids(token)
            def __call__(self, prompt, **kwargs):
                owner.calls.append((tuple(self.tokens), prompt, kwargs))
                return SimpleNamespace(images=[Image.new('RGB', (512, 512), 'orange')], nsfw_content_detected=[False])
        return {'torch': self.torch, 'diffusers': SimpleNamespace(PNDMScheduler=Scheduler, StableDiffusionPipeline=Pipe),
                'custom_attention.unet_2d_condition_custom': SimpleNamespace(UNet2DConditionModel=UNet),
                'experiments.lora_kv_subject_v1.attention': SimpleNamespace(install_subject_lora_kv=
                    lambda *args, **kwargs: owner.installs.append(('lora', args, kwargs)))}

    def test_schema_uses1024_for_every_adapter_and_preserves_empty_fullrank_self_attention(self):
        shapes, empty = snapshots.adapter_schema('fullrank')
        self.assertEqual((len(shapes), len(empty)), (32, 16))
        self.assertTrue(all(shape[1] == 1024 for shape in shapes.values()))
        shapes, empty = snapshots.adapter_schema('lora_r64')
        self.assertEqual((len(shapes), len(empty)), (64, 0))
        self.assertTrue(all(shape == (64, 1024) for key, shape in shapes.items() if '.down.' in key))

    def test_defaults_cover8184_transfers_plus93_stock_and_smoke_is_one_learned_seed42(self):
        total = 0
        for branch in contract.BRANCHES:
            for method in contract.METHODS:
                for step in contract.STEPS:
                    rows = inference.prompt_rows(branch, method, step)
                    self.assertEqual(len(rows), (9 if branch == 'subject' else 12) * 31)
                    self.assertEqual({row['seed'] for row in rows}, set(range(30, 61)))
                    self.assertEqual(len({row['id'] for row in rows}), len(rows))
                    total += len(rows)
                smoke = inference.prompt_rows(branch, method, 2, seeds=(30, 31), smoke=True)
                self.assertEqual(len(smoke), 1)
                self.assertEqual(smoke[0]['seed'], 42)
                self.assertTrue(smoke[0]['condition'].startswith('learned_'))
        self.assertEqual(total, 8184)
        self.assertEqual(len(inference.literal_rows()), 93)
        self.assertEqual(len(inference.literal_rows((42, 43))), 6)
        self.assertEqual((len(inference.literal_rows(smoke=True)), inference.literal_rows(smoke=True)[0]['seed']), (1, 42))
        for seeds in ((), (42, 42), (-1,), (True,)):
            with self.assertRaises(ValueError):
                inference.literal_rows(seeds)

    def test_full_audit_exports_all_four_paired1024_snapshots_without_touching_training(self):
        with patch.dict(sys.modules, {'torch': self.torch}):
            run = self.training_run()
            before = {str(path): self.sha(path) for path in run.rglob('*') if path.is_file()}
            result = snapshots.audit_export(run, 'subject', 'fullrank', self.root / 'exports')
        self.assertEqual(list(result['steps']), [400, 600, 800, 1000])
        self.assertEqual(result['training']['steps'], 1000)
        for step, item in result['steps'].items():
            self.assertEqual((item['step'], item['steps'], item['training_steps'], item['embedding_dim']), (step, step, 1000, 1024))
            embedding = self.registry[str(Path(item['checkpoint']) / '<S*>.bin')]['<S*>']
            self.assertTrue(np.all(embedding.value == step / 1000))
            self.assertEqual(item['final_snapshot_matches_export'], step == 1000)
            snapshots.check_locks(item['hashes'])
        self.assertEqual(before, {str(path): self.sha(path) for path in run.rglob('*') if path.is_file()})

    def test_audit_rejects_wrong_dimension_final_pair_and_ordinary_embedding_drift(self):
        with patch.dict(sys.modules, {'torch': self.torch}):
            for defect in ('width', 'final_pair', 'drift'):
                run = self.training_run('texture', 'lora_r64', True)
                checkpoint = run / 'checkpoints'
                if defect == 'width':
                    self.save({snapshots.shared.EMBEDDING_KEY: Tensor(np.ones((2, 768)))}, checkpoint / 'checkpoint-2/pytorch_model_1.bin')
                elif defect == 'final_pair':
                    self.save({'<T*>': Tensor(np.zeros(1024))}, checkpoint / '<T*>.bin')
                else:
                    path = checkpoint / 'embedding_update_audit.json'
                    audit = json.loads(path.read_text())
                    audit['non_modifier_embedding_drift']['changed_rows'] = 1
                    path.write_text(json.dumps(audit))
                with self.subTest(defect=defect), self.assertRaises(ValueError):
                    snapshots.audit_export(run, 'texture', 'lora_r64', self.root / ('bad-' + defect), True)

    def test_all_six_smokes_export_and_load_only_their_same_step_token_adapter(self):
        with patch.dict(sys.modules, self.modules()):
            for branch in contract.BRANCHES:
                for method in contract.METHODS:
                    self.calls.clear()
                    self.loads.clear()
                    self.installs.clear()
                    run = self.training_run(branch, method, True)
                    output = self.root / 'smoke' / branch / method
                    proof = inference.run(branch, method, run, output, smoke=True, execute=True, device='cpu')
                    self.assertEqual((proof['status'], proof['expected_images'], proof['generated_images']), ('succeeded', 1, 1))
                    self.assertEqual(proof['training_images'], self.data[branch]['rows'])
                    self.assertEqual(len(self.calls), 1)
                    tokens, prompt, kwargs = self.calls[0]
                    self.assertEqual(tokens, (contract.TOKENS[branch],))
                    self.assertEqual(kwargs['generator'].seed, 42)
                    self.assertEqual((kwargs['num_inference_steps'], kwargs['guidance_scale']), (100, 3.5))
                    mask = kwargs['cross_attention_kwargs']['modifier_token_mask']
                    self.assertNotIsInstance(mask, dict)
                    self.assertEqual((int(mask[0].sum()), int(mask[1].sum())), (0, 1))
                    self.assertEqual(self.installs[0][0], 'fullrank' if method == 'fullrank' else 'lora')
                    self.assertEqual(self.loads[0][0], (str(self.model),))
                    self.assertIn('unet', self.loads[0][1])
                    refs = json.loads((output / 'training_references.json').read_text())['images']
                    self.assertEqual(len(refs), self.data[branch]['rows'])
                    self.assertTrue((output / 'training_references_sheet.png').is_file())
                    ledger = json.loads((output / 'step2/generation_status.jsonl').read_text())
                    self.assertEqual(inference.sha256(output / 'step2' / ledger['image_path']), ledger['image_sha256'])
                    self.assertTrue(ledger[branch + '_adapter_loaded'])

    def test_material_texture_three_arm_prompts_and_native_routing_preserve_original_words(self):
        protocol = json.loads(inference.TRANSFER_PROTOCOL.read_text())
        for branch in ('material', 'texture'):
            rows = inference.prompt_rows(branch, 'lora_r64', 400, (42,))
            self.assertEqual(len(rows), 12)
            for obj in protocol['objects']:
                chosen = [row for row in rows if row['label'] == obj]
                self.assertEqual([row['prompt'] for row in chosen], [f'a photo of a {obj}',
                    protocol[branch + '_template'].format(object=obj, attribute=protocol['literal_' + branch]['wood_spoon']),
                    protocol[branch + '_template'].format(object=obj, attribute=contract.TOKENS[branch])])
        with patch.dict(sys.modules, self.modules()):
            run = self.training_run('texture', 'lora_r64')
            proof = inference.run('texture', 'lora_r64', run, self.root / 'texture-full', seeds=(42,), execute=True, device='cpu')
        self.assertEqual(proof['generated_images'], 48)
        self.assertEqual(len(self.loads), 8)
        for tokens, prompt, kwargs in self.calls:
            self.assertEqual(tokens, ('<T*>',) if '<T*>' in prompt else ())
            self.assertEqual('cross_attention_kwargs' in kwargs, bool(tokens))

    def test_literal_controls_are_stock_with_no_override_tokens_or_masks_and_variable_seeds(self):
        with patch.dict(sys.modules, self.modules()):
            proof = inference.run('literal', None, None, self.root / 'literal', seeds=(30, 31), execute=True, device='cpu')
        self.assertEqual((proof['expected_images'], proof['generated_images'], proof['training_images']), (6, 6, 0))
        self.assertEqual(self.installs, [])
        self.assertNotIn('unet', self.loads[0][1])
        self.assertEqual([item.seed for item in self.generators], [30, 31, 30, 31, 30, 31])
        self.assertEqual(len({id(item) for item in self.generators}), 6)
        for tokens, prompt, kwargs in self.calls:
            self.assertEqual(tokens, ())
            self.assertEqual(prompt, 'a photo of a spoon')
            self.assertNotIn('cross_attention_kwargs', kwargs)
            self.assertNotIn('negative_prompt', kwargs)

    def test_loader_rejects_sd15_width_or_missing_linear_projection_before_generation(self):
        with patch.dict(sys.modules, self.modules()):
            for defect in ('width', 'linear'):
                self.pipeline_defect = defect
                with self.subTest(defect=defect), self.assertRaisesRegex(ValueError, '1024-dimensional'):
                    inference.load_pipeline([], 'cpu', model_dir=self.model, stock=True)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
