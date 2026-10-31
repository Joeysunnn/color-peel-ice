"""CPU evidence for a stock Diffusers pipeline and the fixed15-image CFG grid."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.sd15_native_spoon_cfg_v1 import run as native


class NativeSpoonCfgTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='native-spoon-cfg-', dir=os.environ.get('TMP'))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'weights.bin').write_bytes(b'frozen stock SD1.5 weights')
        (self.model / 'checker.json').write_text('{"enabled":true}')
        self.spec = {'root': str(self.model), 'revision': 'frozen-sd15', 'repo': 'stock-SD1.5'}
        receipt = {'model': self.spec, 'safety_checker_source': {'root': 'existing shared checker'},
                   'files': {name: native.sha256(self.model / name) for name in ('weights.bin', 'checker.json')}}
        (self.model / 'model_source.json').write_text(json.dumps(receipt))
        self.protocol = self.root / 'protocol.json'
        self.protocol.write_text(json.dumps({'model': self.spec}))
        self.git = {'commit': 'stock-test', 'branch': 'codex/stock-test'}
        for item in (patch.dict(os.environ, {'COLORPEEL_OUTPUT_ROOT': str(self.root)}),
                     patch.object(native, 'PROTOCOL', self.protocol),
                     patch.object(native, 'PREVIOUS', self.root / 'no_previous_baseline'),
                     patch.object(native, 'git_info', return_value=self.git),
                     patch.object(native, 'package_versions', return_value={'torch': 'fake', 'diffusers': 'fake'})):
            item.start()
            self.addCleanup(item.stop)
        self.calls, self.generators, self.pipeline_loads, self.scheduler_loads, self.moves = [], [], [], [], []

    def modules(self, defect=None, result_defect=None):
        owner = self

        class Generator:
            def __init__(self, device):
                self.device, self.seed = device, None
                owner.generators.append(self)

            def manual_seed(self, seed):
                self.seed = seed
                return self

        class PNDMScheduler:
            __module__ = 'diffusers.schedulers.scheduling_pndm'

            def __init__(self):
                self.config = {'num_train_timesteps': 1000, 'skip_prk_steps': True}

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                owner.scheduler_loads.append((args, kwargs))
                return cls()

        class AttnProcessor:
            __module__ = 'diffusers.models.attention_processor'

        class UNet2DConditionModel:
            __module__ = 'diffusers.models.unet_2d_condition'
            dtype = 'float16'

            def __init__(self):
                processor = AttnProcessor()
                if defect == 'custom_attention':
                    processor = type('TokenLocalKVAttnProcessor', (), {'__module__': 'custom_attention.attention_processor_custom'})()
                elif defect == 'lora_attention':
                    processor = type('LoRAAttnProcessor', (), {'__module__': 'diffusers.models.attention_processor'})()
                self.attn_processors = {'block.attn2.processor': processor}

            def load_attn_procs(self, *args, **kwargs):
                raise AssertionError('stock test must never load adapter K/V')

            def set_attn_processor(self, *args, **kwargs):
                raise AssertionError('stock test must never replace attention processors')

        class StableDiffusionPipeline:
            __module__ = 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion'

            def __init__(self, scheduler):
                self.scheduler = object() if defect == 'scheduler' else scheduler
                self.safety_checker = None if defect == 'checker' else object()
                self.unet = UNet2DConditionModel()
                if defect == 'custom_unet':
                    self.unet = type('CustomUNet', (UNet2DConditionModel,), {'__module__': 'custom_attention.unet_2d_condition_custom'})()
                vocab = {'a': 1, 'photo': 2, 'of': 3, 'spoon': 4}
                if defect in native.TOKENS:
                    vocab[defect] = 5
                self.tokenizer = SimpleNamespace(get_vocab=lambda: vocab)

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                owner.pipeline_loads.append((args, kwargs))
                return cls(kwargs['scheduler'])

            def to(self, device):
                owner.moves.append(device)
                return self

            def set_progress_bar_config(self, **kwargs):
                owner.assertEqual(kwargs, {'disable': True})

            def load_textual_inversion(self, *args, **kwargs):
                raise AssertionError('stock test must never load learned embeddings')

            def load_lora_weights(self, *args, **kwargs):
                raise AssertionError('stock test must never load LoRA')

            def __call__(self, prompt, **kwargs):
                owner.calls.append((prompt, kwargs))
                mode = 'L' if result_defect == 'mode' else 'RGB'
                size = (256, 256) if result_defect == 'size' else (512, 512)
                image = Image.new(mode, size)
                flags = None if result_defect == 'missing_safety' else [kwargs['generator'].seed == 42]
                if result_defect == 'multiple_safety':
                    flags = [False, False]
                images = [image, image] if result_defect == 'multiple_images' else [image]
                return SimpleNamespace(images=images, nsfw_content_detected=flags)

        if defect == 'custom_pipeline':
            StableDiffusionPipeline.__module__ = 'project.custom_pipeline'
        return {'torch': SimpleNamespace(float16='float16', Generator=Generator,
                    cuda=SimpleNamespace(get_device_name=lambda device: 'fake GPU ' + device)),
                'diffusers': SimpleNamespace(StableDiffusionPipeline=StableDiffusionPipeline,
                                            PNDMScheduler=PNDMScheduler)}

    def test_grid_has15_unique_images_with_only_the_requested_prompt_cfgs_and_seeds(self):
        rows = native.rows()
        self.assertEqual(len(rows), 15)
        self.assertEqual({(row['cfg'], row['seed']) for row in rows},
                         {(cfg, seed) for cfg in (3.5, 5.0, 7.5) for seed in range(42, 47)})
        self.assertEqual(len({row['id'] for row in rows}), 15)
        self.assertEqual(len({row['image_path'] for row in rows}), 15)
        self.assertEqual({row['prompt'] for row in rows}, {'a photo of a spoon'})
        self.assertTrue(all(not Path(row['image_path']).is_absolute() and '..' not in Path(row['image_path']).parts for row in rows))

    def test_stock_loader_and_scheduler_generate15_checked_pngs_with_fresh_seeded_generators(self):
        output = self.root / 'executed'
        with patch.dict(sys.modules, self.modules()):
            proof = native.run(output, 'cuda:2', True)
        self.assertEqual(self.scheduler_loads, [((str(self.model),), {'subfolder': 'scheduler', 'local_files_only': True})])
        self.assertEqual(len(self.pipeline_loads), 1)
        args, kwargs = self.pipeline_loads[0]
        self.assertEqual(args, (str(self.model),))
        self.assertNotIn('unet', kwargs)
        self.assertEqual(set(kwargs), {'scheduler', 'torch_dtype', 'local_files_only', 'low_cpu_mem_usage'})
        self.assertEqual((kwargs['torch_dtype'], kwargs['local_files_only'], kwargs['low_cpu_mem_usage']), ('float16', True, False))
        self.assertEqual(self.moves, ['cuda:2'])
        self.assertEqual(len(self.calls), 15)
        self.assertEqual(len({id(generator) for generator in self.generators}), 15)
        for row, (prompt, arguments), generator in zip(native.rows(), self.calls, self.generators):
            self.assertEqual(prompt, 'a photo of a spoon')
            self.assertIs(arguments['generator'], generator)
            self.assertEqual((generator.device, generator.seed), ('cuda:2', row['seed']))
            self.assertEqual((arguments['guidance_scale'], arguments['num_inference_steps'], arguments['width'], arguments['height']),
                             (row['cfg'], 100, 512, 512))
            self.assertNotIn('negative_prompt', arguments)
            self.assertNotIn('cross_attention_kwargs', arguments)
        for seed in range(42, 47):
            seeded = [generator for generator in self.generators if generator.seed == seed]
            self.assertEqual(len(seeded), 3)
            self.assertEqual(len({id(generator) for generator in seeded}), 3)
        self.assertEqual((proof['status'], proof['generated_images'], proof['safety_filtered']), ('succeeded', 15, 3))
        self.assertFalse(proof['adapters'])
        self.assertEqual(proof['learned_tokens'], [])
        self.assertTrue(proof['runtime']['pipeline_class'].startswith('diffusers.'))
        self.assertTrue(proof['runtime']['unet_class'].startswith('diffusers.'))
        self.assertTrue(all(name.startswith('diffusers.') for name in proof['runtime']['attention_processors']))
        self.assertEqual(proof['runtime']['dtype'], 'float16')
        ledger = [json.loads(line) for line in (output / 'generation_status.jsonl').read_text().splitlines()]
        self.assertEqual(ledger, proof['records'])
        self.assertEqual(len(ledger), 15)
        for row in ledger:
            image = output / row['image_path']
            self.assertEqual(native.sha256(image), row['image_sha256'])
            self.assertFalse(row['subject_adapter_loaded'])
            self.assertFalse(row['texture_adapter_loaded'])
            with Image.open(image) as actual:
                self.assertEqual((actual.format, actual.mode, actual.size), ('PNG', 'RGB', (512, 512)))
        with Image.open(output / 'contact_sheet.png') as sheet:
            self.assertEqual(sheet.size, (1152, 2180))
        self.assertEqual((output / 'index.html').read_text(encoding='utf8').count('<tr><th>seed'), 5)

    def test_dry_run_records_the_grid_without_loading_torch_pipeline_or_scheduler(self):
        output = self.root / 'dry_run'
        with patch.dict(sys.modules, self.modules()):
            proof = native.run(output)
        self.assertEqual(proof['status'], 'dry_run_verified')
        self.assertEqual(proof['records'], [])
        self.assertEqual(len((output / 'generation_manifest.jsonl').read_text().splitlines()), 15)
        self.assertEqual(self.pipeline_loads, [])
        self.assertEqual(self.scheduler_loads, [])
        self.assertEqual(self.generators, [])
        self.assertFalse((output / 'generation_status.jsonl').exists())

    def test_custom_classes_lora_tokens_missing_checker_and_wrong_scheduler_fail_before_generation(self):
        defects = ('custom_pipeline', 'custom_unet', 'custom_attention', 'lora_attention', 'checker', 'scheduler', *native.TOKENS)
        for index, defect in enumerate(defects):
            output = self.root / f'rejected_runtime_{index}'
            with self.subTest(defect=defect), patch.dict(sys.modules, self.modules(defect=defect)):
                with self.assertRaisesRegex(ValueError, 'require stock'):
                    native.run(output, 'cuda:0', True)
            self.assertEqual(self.calls, [])
            proof = json.loads((output / 'provenance.json').read_text())
            self.assertEqual(proof['status'], 'failed')
            self.assertFalse((output / 'generation_status.jsonl').exists())

    def test_missing_or_ambiguous_safety_and_non512_rgb_results_are_not_saved(self):
        for defect in ('missing_safety', 'multiple_safety', 'multiple_images', 'size', 'mode'):
            output = self.root / ('rejected_result_' + defect)
            with self.subTest(defect=defect), patch.dict(sys.modules, self.modules(result_defect=defect)):
                with self.assertRaisesRegex(ValueError, 'explicit safety result'):
                    native.run(output, 'cuda:0', True)
            proof = json.loads((output / 'provenance.json').read_text())
            self.assertEqual((proof['status'], proof['records']), ('failed', []))
            self.assertFalse((output / 'images').exists())
            self.assertFalse((output / 'generation_status.jsonl').exists())

    def test_changed_frozen_model_is_rejected_before_pipeline_loading(self):
        (self.model / 'weights.bin').write_bytes(b'changed model weights')
        output = self.root / 'changed_model'
        with patch.dict(sys.modules, self.modules()), self.assertRaisesRegex(ValueError, 'frozen model or source bytes changed'):
            native.run(output, 'cuda:0', True)
        self.assertEqual(self.pipeline_loads, [])
        self.assertEqual(self.scheduler_loads, [])
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
