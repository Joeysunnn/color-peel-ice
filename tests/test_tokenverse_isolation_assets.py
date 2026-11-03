"""CPU-only bank identity/length tests and a mocked filtered generation stop."""

from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.sd21_tokenverse_isolation_v1 import assets


class Tokenizer:
    model_max_length = 77

    def encode(self, prompt, add_special_tokens=True):
        return list(range(len(prompt.split()) + (2 if add_special_tokens else 0)))


class IsolationBankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bank = self.root / 'bank'
        (self.bank / 'images').mkdir(parents=True)
        self.records = []
        for index, caption in enumerate(assets.CAPTIONS):
            path = self.bank / 'images' / f'{index:02d}.png'
            Image.new('RGB', (512, 512), (index + 1, 70, 120)).save(path)
            self.records.append({'id': index, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'caption': caption, 'size': [512, 512], 'mode': 'RGB', 'seed': 1000 + index, 'status': 'ok'})
        self.manifest = {'schema': 1, 'status': 'ready', 'sampling': assets.SAMPLING,
                         'model_receipt': {'model': 'synthetic test receipt'}, 'git': {'commit': 'fixture'}, 'records': self.records}
        self.path = self.bank / 'manifest.json'
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.manifest), encoding='utf8')

    def test_exact_caption_bank_and_image_locks(self):
        self.assertEqual(len(assets.CAPTIONS), 25)
        self.assertEqual(len(set(assets.CAPTIONS)), 25)
        for caption in assets.CAPTIONS:
            self.assertLessEqual(len(caption.split()), 12)
            self.assertTrue(set(caption.lower().split()).isdisjoint(
                {'mailbox', 'orange', 'wood', 'texture', 'street', 'snow', 'flowers', 'beach'}))
        locks = assets.verify_bank(self.path)
        self.assertEqual(len(locks), 26)
        for filename, digest in locks.items():
            self.assertEqual(hashlib.sha256(Path(filename).read_bytes()).hexdigest(), digest)

    def test_duplicate_missing_caption_seed_or_status_rejected(self):
        original = copy.deepcopy(self.manifest)
        for field, bad in (('id', 1), ('caption', 'wrong caption'), ('seed', 42), ('status', 'safety_filtered')):
            self.manifest = copy.deepcopy(original)
            self.manifest['records'][0][field] = bad
            self.write()
            with self.subTest(field=field), self.assertRaises(ValueError):
                assets.verify_bank(self.path)
        self.manifest = copy.deepcopy(original)
        self.manifest['records'].pop()
        self.write()
        with self.assertRaises(ValueError):
            assets.verify_bank(self.path)

    def test_sha_black_image_and_extra_image_rejected(self):
        path = Path(self.records[0]['path'])
        Image.new('RGB', (512, 512), (20, 30, 40)).save(path)
        with self.assertRaisesRegex(ValueError, 'SHA'):
            assets.verify_bank(self.path)
        Image.new('RGB', (512, 512), 0).save(path)
        self.records[0]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write()
        with self.assertRaisesRegex(ValueError, 'nonblack'):
            assets.verify_bank(self.path)
        Image.new('RGB', (512, 512), (20, 30, 40)).save(path)
        self.records[0]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write()
        Image.new('RGB', (512, 512), 'white').save(self.bank / 'images' / 'extra.png')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            assets.verify_bank(self.path)

    def test_complete_prompt_lengths_include_special_tokens(self):
        from src.train.concept_isolation import combine_prompt
        prompts = ['a photo of <S*> mailbox', 'a cube with <T*> texture']
        result = assets.preflight(Tokenizer(), prompts)
        self.assertEqual(result['combined_pairs_checked'], 50)
        self.assertEqual(result['combined_lengths'][0][0], len(combine_prompt(prompts[0], assets.CAPTIONS[0]).split()) + 2)
        self.assertFalse(result['modifier_tokens_added'])
        for count, allowed in ((77, True), (78, False)):
            tokenizer = SimpleNamespace(model_max_length=77, encode=lambda *args, **kwargs: list(range(count)))
            if allowed:
                assets.preflight(tokenizer, prompts)
            else:
                with self.assertRaisesRegex(ValueError, 'CLIP77'):
                    assets.preflight(tokenizer, prompts)

    def test_filtered_generation_stops_without_resampling_or_negative_prompt(self):
        from experiments.sd21_wood_branches_v1 import inference as shared
        output, model = self.root / 'generated', self.root / 'model'
        model.mkdir()
        (model / 'model_source.json').write_text('{"model":"fixture"}')
        calls = []

        class Pipeline:
            tokenizer = Tokenizer()

            def set_progress_bar_config(self, **kwargs):
                pass

            def __call__(self, prompt, **kwargs):
                calls.append((prompt, kwargs))
                return SimpleNamespace(images=[Image.new('RGB', (512, 512), 'white')],
                                       nsfw_content_detected=[len(calls) == 2])

        torch = SimpleNamespace(no_grad=nullcontext, cuda=SimpleNamespace(empty_cache=lambda: None),
                                Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: seed))
        with patch.dict(sys.modules, {'torch': torch}), patch.object(shared, 'ROOT', self.root / 'code'), \
                patch.object(shared, 'MODEL_DIR', model), patch.object(shared, 'git_info', return_value={'commit': 'fixture'}), \
                patch.object(shared, 'load_pipeline', return_value=Pipeline()) as load:
            with self.assertRaisesRegex(ValueError, 'no replacement'):
                assets.generate(output, ['a photo of <S*> mailbox'])
        load.assert_called_once_with([], 'cuda:0', stock=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual([item[1]['generator'] for item in calls], [1000, 1001])
        self.assertTrue(all('negative_prompt' not in item[1] for item in calls))
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'safety_filtered')
        self.assertEqual(len(manifest['records']), 2)
        self.assertTrue((output / 'contact_sheet.png').is_file())
        self.assertTrue((output / 'index.html').is_file())
        with self.assertRaisesRegex(ValueError, 'ready'):
            assets.verify_bank(output / 'manifest.json')

    def test_ready_generation_returns_final_file_locks_and_25_actual_prompts(self):
        from experiments.sd21_wood_branches_v1 import inference as shared
        output, model = self.root / 'generated', self.root / 'model'
        model.mkdir()
        (model / 'model_source.json').write_text('{"model":"fixture"}')
        calls = []

        class Pipeline:
            tokenizer = Tokenizer()

            def set_progress_bar_config(self, **kwargs):
                pass

            def __call__(self, prompt, **kwargs):
                calls.append((prompt, kwargs))
                return SimpleNamespace(images=[Image.new('RGB', (512, 512), 'white')], nsfw_content_detected=[False])

        torch = SimpleNamespace(no_grad=nullcontext, cuda=SimpleNamespace(empty_cache=lambda: None),
                                Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: seed))
        with patch.dict(sys.modules, {'torch': torch}), patch.object(shared, 'ROOT', self.root / 'code'), \
                patch.object(shared, 'MODEL_DIR', model), patch.object(shared, 'git_info', return_value={'commit': 'fixture'}), \
                patch.object(shared, 'load_pipeline', return_value=Pipeline()):
            result = assets.generate(output, ['a cube with <T*> texture'])
        self.assertEqual([prompt for prompt, kwargs in calls], list(assets.CAPTIONS))
        self.assertEqual([kwargs['generator'] for prompt, kwargs in calls], list(range(1000, 1025)))
        self.assertTrue(all(kwargs['num_inference_steps'] == 100 and kwargs['guidance_scale'] == 3.5
                            and kwargs['width'] == kwargs['height'] == 512 and 'negative_prompt' not in kwargs
                            for prompt, kwargs in calls))
        self.assertEqual(result['count'], 25)
        for filename, digest in result['source_locks'].items():
            self.assertEqual(hashlib.sha256(Path(filename).read_bytes()).hexdigest(), digest)
        self.assertEqual(len(assets.verify_bank(result['manifest_path'])), 26)


if __name__ == '__main__':
    unittest.main()
