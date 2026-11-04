"""Selected-branch initialization preserves uncertainty and bounded retry evidence."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from tests.methods.colorpeel_ice.test_vlm_initialization import Tokenizer, response
from src.methods.colorpeel_ice import vlm_initialization as vlm
from src.train.initializer_token_utils import (
    SELECTED_SCHEMA, load_initializer_manifest, validate_initializer_training,
)


class SelectedInitializerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / 'input.png'
        Image.new('RGB', (32, 24), (30, 60, 90)).save(self.image)
        self.sd = self.root / 'sd'
        (self.sd / 'tokenizer').mkdir(parents=True)
        (self.sd / 'tokenizer/vocab.json').write_text('{}')
        self.output = self.root / 'output'
        self.calls = []

    def run_fixture(self, responses, selected, allow_partial=False):
        calls = self.calls
        class Predictor:
            runtime = {'dtype': 'float16', 'do_sample': False}
            def __call__(self, image):
                calls.append(self.prompt)
                return json.dumps(responses[len(calls) - 1])
        transformers = SimpleNamespace(CLIPTokenizer=SimpleNamespace(
            from_pretrained=lambda *a, **k: Tokenizer()))
        with patch.object(vlm, 'model_receipt', return_value={'metadata_files': {}, 'weights': []}), \
                patch.object(vlm, 'QwenPredictor', return_value=Predictor()), \
                patch.dict(sys.modules, {'transformers': transformers}):
            return vlm.run(self.image, self.root / 'qwen', self.sd, self.output,
                           device='cpu', selected_branches=selected, allow_partial=allow_partial)

    def test_color_only_unknown_texture_succeeds_without_guessing_or_retry(self):
        value = response()
        value['texture'].update(confidence='unknown', candidates=[])
        path = self.run_fixture([value], ['color'])
        manifest = load_initializer_manifest(path, Tokenizer(), self.sd)
        self.assertEqual(manifest['schema'], SELECTED_SCHEMA)
        self.assertEqual(manifest['selected_branches'], ['color'])
        self.assertEqual(manifest['selection_attempts'], 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(manifest['response'], value)
        self.assertEqual(set(manifest['branches']), {'subject', 'color', 'material', 'texture'})
        self.assertIsNone(manifest['branches']['texture']['initializer'])
        self.assertEqual(manifest['branches']['color']['initializer'], 'orange')

    def test_all_four_unknown_texture_fails_without_second_query(self):
        value = response()
        value['texture'].update(confidence='unknown', candidates=[])
        with self.assertRaisesRegex(ValueError, 'unknown semantic confidence'):
            self.run_fixture([value], list(vlm.TOKENS))
        self.assertEqual(len(self.calls), 1)
        self.assertFalse((self.output / 'initializers.json').exists())
        self.assertEqual(json.loads((self.output / 'status.json').read_text())['status'], 'failed')
        self.assertEqual(json.loads((self.output / 'attempts/1/raw_response.txt').read_text()), value)
        self.assertFalse((self.output / 'attempts/2').exists())

    def test_bpe_retry_preserves_both_attempts_and_rejects_prior_evidence_mutation(self):
        invalid, valid = response(), response()
        invalid['texture']['candidates'] = ['speckled']
        path = self.run_fixture([invalid, valid], ['texture'])
        manifest = load_initializer_manifest(path, Tokenizer(), self.sd)
        self.assertEqual(manifest['selection_attempts'], 2)
        self.assertEqual(manifest['branches']['texture']['initializer'], 'plain')
        self.assertEqual(len(self.calls), 2)
        self.assertIn('multiple_bpe', self.calls[1])
        self.assertIn('do not change visual evidence', self.calls[1])
        for attempt, expected in ((1, invalid), (2, valid)):
            folder = self.output / 'attempts' / str(attempt)
            self.assertEqual(json.loads((folder / 'raw_response.txt').read_text()), expected)
            for name in ('prompt.txt', 'raw_response.txt', 'generation_receipt.json'):
                self.assertIn(str(folder / name), manifest['source_locks'])
        (self.output / 'attempts/1/raw_response.txt').write_text('changed failed evidence')
        with self.assertRaises(ValueError):
            load_initializer_manifest(path, Tokenizer(), self.sd)

    def test_two_failed_bpe_attempts_do_not_start_a_third_or_write_manifest(self):
        invalid = response()
        invalid['texture']['candidates'] = ['speckled']
        with self.assertRaisesRegex(ValueError, 'no valid single-BPE'):
            self.run_fixture([invalid, invalid], ['texture'])
        self.assertEqual(len(self.calls), 2)
        self.assertFalse((self.output / 'initializers.json').exists())
        self.assertTrue((self.output / 'attempts/1/raw_response.txt').exists())
        self.assertTrue((self.output / 'attempts/2/raw_response.txt').exists())

    def test_unselected_training_branch_is_rejected_even_if_its_initializer_is_valid(self):
        path = self.run_fixture([response()], ['color'])
        manifest = load_initializer_manifest(path, Tokenizer(), self.sd)
        self.assertEqual(manifest['branches']['material']['initializer'], 'glossy')
        with self.assertRaisesRegex(ValueError, 'not selected'):
            validate_initializer_training(SimpleNamespace(modifier_token='<M*>',
                                                           initializer_token='glossy'), manifest)

    def test_legacy_default_keeps_strict_all_four_and_never_retries(self):
        invalid = response()
        invalid['texture']['candidates'] = ['speckled']
        with self.assertRaisesRegex(ValueError, 'no valid single-BPE'):
            self.run_fixture([invalid, response()], None)
        self.assertEqual(self.calls, [vlm.PROMPT])
        self.assertFalse((self.output / 'attempts').exists())
        self.assertFalse((self.output / 'initializers.json').exists())

    def test_opt_in_partial_manifest_authorizes_only_available_requested_branches(self):
        value = response()
        value['texture'].update(confidence='unknown', candidates=[])
        path = self.run_fixture([value], list(vlm.TOKENS), allow_partial=True)
        manifest = load_initializer_manifest(path, Tokenizer(), self.sd)
        self.assertEqual(manifest['status'], 'verified')
        self.assertEqual(manifest['requested_branches'], list(vlm.TOKENS))
        self.assertEqual(manifest['selected_branches'], ['subject', 'color', 'material'])
        self.assertTrue(any('unknown semantic' in reason for reason in manifest['unavailable_branches']))
        self.assertEqual(manifest['response'], value)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(json.loads((self.output / 'status.json').read_text())['status'], 'partial')
        with self.assertRaisesRegex(ValueError, 'not selected'):
            validate_initializer_training(SimpleNamespace(modifier_token='<T*>',
                                                           initializer_token=None), manifest)

    def test_partial_after_bpe_repair_exhaustion_preserves_failure_and_both_attempts(self):
        invalid = response()
        invalid['texture']['candidates'] = ['speckled']
        path = self.run_fixture([invalid, invalid], list(vlm.TOKENS), allow_partial=True)
        manifest = load_initializer_manifest(path, Tokenizer(), self.sd)
        self.assertEqual(manifest['selection_attempts'], 2)
        self.assertNotIn('texture', manifest['selected_branches'])
        self.assertEqual(manifest['unavailable_branches'], ['texture: no valid single-BPE initializer'])
        for index in (1, 2):
            self.assertIn(str(self.output / f'attempts/{index}/raw_response.txt'), manifest['source_locks'])

    def test_partial_with_no_available_requested_branch_fails(self):
        value = response()
        value['texture'].update(confidence='unknown', candidates=[])
        with self.assertRaisesRegex(ValueError, 'unknown semantic'):
            self.run_fixture([value], ['texture'], allow_partial=True)
        self.assertFalse((self.output / 'initializers.json').exists())

    def test_partial_flag_cannot_relax_legacy_implicit_all_four_mode(self):
        with self.assertRaisesRegex(ValueError, 'explicit selected branches'):
            self.run_fixture([response()], None, allow_partial=True)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
