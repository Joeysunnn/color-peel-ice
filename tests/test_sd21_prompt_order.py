from collections import Counter
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.sd21_prompt_order_v1 import conditioning
from experiments.sd21_prompt_order_v1 import run as study

try:
    import torch
except ImportError:
    torch = None


class PlanTests(unittest.TestCase):
    def test_exact12_cases_preserve_words_and_pin_weights(self):
        rows = study.plan()
        self.assertEqual(rows, study.plan('prefix'))
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row['image_path'] for row in rows}), 12)
        self.assertEqual({row['condition'] for row in rows[:6]}, {'P0_suffix'})
        self.assertEqual({row['condition'] for row in rows[6:]}, {'P1_prefix'})
        for row in rows:
            self.assertEqual(row['method'], 'lora_r64')
            self.assertEqual(row['steps'], {'subject': 1000, 'color': 150, 'texture': 400})
            self.assertIn(row['seed'], (31, 38))
            suffix = study.sct.SCENES[row['scene']]
            expected = study.sct.PROMPT + ' ' + suffix
            self.assertEqual(Counter(row['prompt'].split()), Counter(expected.split()))
            self.assertEqual(row['prompt'], expected if row['condition'] == 'P0_suffix' else suffix + ' ' + study.sct.PROMPT)
            self.assertTrue(all(row['prompt'].count(token) == 1 for token in study.sct.TOKENS.values()))

    def test_surface_only_changes_the_ordinary_word_after_texture(self):
        original, rows = study.plan(), study.plan('surface')
        self.assertEqual(rows[:6], original[:6])
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row['image_path'] for row in rows}), 12)
        for baseline, changed in zip(rows[:6], rows[6:]):
            self.assertEqual(changed['condition'], 'P1_surface')
            self.assertEqual(changed['prompt'], baseline['prompt'].replace(' with <T*> texture ', ' with <T*> surface '))
            for key in ('method', 'scene', 'seed', 'steps'):
                self.assertEqual(changed[key], baseline[key])
        with self.assertRaises(ValueError):
            study.plan('short')
        with self.assertRaises(ValueError):
            conditioning.surface_body('a photo of <T*> texture in the snow')

    def records(self):
        return [{**row, 'baseline_sha_equal': True, 'baseline_status_equal': True} for row in study.plan()[:6]]

    def test_prefix_requires_all_six_exact_baselines(self):
        good = self.records()
        study.require_baselines(good)
        for bad in (good[:5], good[:5] + [good[0]], [{**row, 'baseline_sha_equal': False} for row in good],
                    [{**row, 'baseline_status_equal': False} for row in good]):
            with self.subTest(records=bad), self.assertRaises(ValueError):
                study.require_baselines(bad)

    def test_original_screen_rows_and_six_image_locks_are_checked(self):
        ledger = [{**row, 'image_sha256': 'verified', 'status': 'ok'} for row in study.screen.plan()]
        root = (Path.cwd() / 'immutable_screen_fixture').resolve()
        with patch.object(study.sct, 'sha256', return_value='verified'):
            baselines, locks = study.screen_baselines(root, ledger)
            self.assertEqual(len(baselines), 6)
            self.assertEqual(len(locks), 6)
            bad = copy.deepcopy(ledger)
            next(row for row in bad if row['method'] == 'lora_r64' and row['condition'] == 't400')['prompt'] += ' extra'
            with self.assertRaises(ValueError):
                study.screen_baselines(root, bad)
            with self.assertRaises(ValueError):
                study.screen_baselines(root, ledger[:-1])
        with patch.object(study.sct, 'sha256', return_value='changed'):
            with self.assertRaises(ValueError):
                study.screen_baselines(root, ledger)

    def test_prepare_locks_screen_audit_status_ledger_and_preserves_training_sources(self):
        saved = {key: {key: 'unchanged'} for key in ('sources', 'texture_sources', 'training_gits', 'references', 'texture_conditions')}
        saved.update(source_locks={}, previous_source_audit='/old_scene/source_audit.json', inference_git='screen_git')
        current = {**copy.deepcopy(saved), 'inference_git': 'new_git'}
        status = {'status': 'succeeded', 'expected_images': 96, 'generated_images': 96, 'inference_git': 'screen_git',
                  'sampling': {**study.sct.st.shared.inference.SAMPLING, 'seeds': [31, 38]}}
        path = (Path.cwd() / 'screen_fixture/source_audit.json').resolve()
        with patch.object(study.screen, 'prepare', return_value=current), \
             patch.object(study.sct.st, 'read_json', side_effect=[saved, status]), \
             patch.object(Path, 'read_text', return_value='{}\n'), \
             patch.object(study, 'screen_baselines', return_value=({'street:31': {}}, {'image': 'image_hash'})), \
             patch.object(study.sct.st.shared.snapshots, 'check_locks'), \
             patch.object(study.sct, 'sha256', return_value='file_hash'):
            result = study.prepare(path)
        self.assertEqual(result['screen_inference_git'], 'screen_git')
        self.assertEqual(result['inference_git'], 'new_git')
        self.assertTrue(all(str(path.parent / name) in result['source_locks']
                            for name in ('source_audit.json', 'status.json', 'generation_status.jsonl')))
        self.assertEqual(result['sources'], saved['sources'])
        self.assertEqual(saved['source_locks'], {})

    def test_multi_token_mailbox_is_recorded_not_silently_truncated(self):
        tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: [10, 11] if text == 'mailbox' else [12])
        result = conditioning.initializer_ids(tokenizer)
        self.assertFalse(result['same_length_replacement_available'])
        self.assertEqual(result['ids']['subject'], [10, 11])


@unittest.skipIf(torch is None, 'requires server torch')
class TensorTests(unittest.TestCase):
    def test_prompt_specific_cfg_masks_reject_stale_positions(self):
        mapping = {'<S*>': 3, '<C*>': 4, '<T*>': 5}
        tokenizer = SimpleNamespace(convert_tokens_to_ids=mapping.__getitem__)
        ids = torch.zeros(1, 77, dtype=torch.long)
        ids[0, 4:7] = torch.tensor([3, 4, 5])
        masks = {branch: torch.cat((torch.zeros_like(ids, dtype=torch.bool), ids == mapping[token]))
                 for branch, token in study.sct.TOKENS.items()}
        conditioning.validate_gates(ids, {'modifier_token_mask': masks}, tokenizer, study.sct.TOKENS)
        bad = {key: value.clone() for key, value in masks.items()}
        bad['subject'][0, 4] = True
        with self.assertRaises(ValueError):
            conditioning.validate_gates(ids, {'modifier_token_mask': bad}, tokenizer, study.sct.TOKENS)
        shifted = torch.roll(ids, 3, dims=1)
        with self.assertRaises(ValueError):
            conditioning.validate_gates(shifted, {'modifier_token_mask': masks}, tokenizer, study.sct.TOKENS)

    def test_latent_hook_observes_only_first_forward_without_mutating_rng(self):
        class Unet(torch.nn.Module):
            def forward(self, sample, timestep):
                return sample + 1
        module, captured = Unet(), {}
        sample = torch.randn(2, 4, 64, 64)
        state = torch.get_rng_state().clone()
        handle = conditioning.capture_initial_input(module, captured)
        output = module(sample, torch.tensor(999))
        module(sample + 5, torch.tensor(998))
        handle.remove()
        self.assertTrue(torch.equal(output, sample + 1))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(captured['sha256'], conditioning.tensor_sha256(sample[:1]))
        self.assertEqual(captured['timestep'], 999)

    def make_pipe(self, causal=True, split_surface=False):
        class Tokenizer:
            model_max_length = 77
            def __init__(self):
                self.vocab = {token: i + 10 for i, token in enumerate(study.sct.TOKENS.values())}
                self.vocab.update(mailbox=20, orange=21)
            def encode(self, text, add_special_tokens=True):
                words = text.replace('surface', 'sur face').split() if split_surface else text.split()
                ids = [self.vocab.setdefault(word, len(self.vocab) + 30) for word in words]
                return [1] + ids + [2] if add_special_tokens else ids
            def convert_tokens_to_ids(self, token):
                return self.vocab[token]
            def convert_ids_to_tokens(self, ids):
                inverse = {value: key for key, value in self.vocab.items()}
                inverse.update({0: '[pad]', 1: '[bos]', 2: '[eos]'})
                return [inverse[value] for value in ids]
            def __call__(self, text, **kwargs):
                ids = self.encode(text)
                return SimpleNamespace(input_ids=torch.tensor([ids + [0] * (77 - len(ids))]),
                                       attention_mask=torch.tensor([[1] * len(ids) + [0] * (77 - len(ids))]))
        class Encoder(torch.nn.Module):
            config = SimpleNamespace(use_attention_mask=False)
            def forward(self, ids):
                values = ids.float().cumsum(1)
                if not causal:
                    values = values + ids.float().sum(1, keepdim=True)
                return (values.unsqueeze(-1).expand(-1, -1, 4),)
        tokenizer, encoder = Tokenizer(), Encoder().eval()
        pipe = SimpleNamespace(tokenizer=tokenizer, text_encoder=encoder,
            unet=SimpleNamespace(device='cpu', attn_processors={}), maybe_convert_prompt=lambda p, t: p)
        pipe._encode_prompt = lambda prompt, *args: encoder(tokenizer(prompt).input_ids)[0]
        return pipe

    def test_actual_token_diagnostic_uses_same_slot_counterfactual(self):
        pipe = self.make_pipe()
        result = conditioning.diagnose(pipe, study.sct.PROMPT, study.sct.SCENES, study.sct.TOKENS)
        self.assertEqual(result['mode'], 'prefix')
        self.assertTrue(all(value == 0 for row in result['suffix_modifier_hidden_max_abs_vs_body'].values() for value in row.values()))
        self.assertTrue(all(row['texture_hidden_max_abs'] > 0 for row in result['same_length_SC_initializer_counterfactual'].values()))
        self.assertEqual(len(result['prompts']), 7)

    def test_surface_proves_modifier_identity_and_records_downstream_bpe_changes(self):
        for split in (False, True):
            with self.subTest(split_surface=split):
                result = conditioning.diagnose(self.make_pipe(split_surface=split), study.sct.PROMPT,
                                              study.sct.SCENES, study.sct.TOKENS, mode='surface')
                self.assertEqual(result['mode'], 'surface')
                self.assertEqual(len(result['prompts']), 7)
                for row in result['surface_after_texture_audit'].values():
                    self.assertTrue(row['ids_through_texture_equal'])
                    self.assertTrue(row['modifier_positions_equal'])
                    self.assertEqual(row['modifier_hidden_equal'], {branch: True for branch in study.sct.TOKENS})
                    self.assertEqual(row['surface_bpe_length'] - row['suffix_bpe_length'], int(split))
                    changes = row['downstream_positional_changes']
                    self.assertTrue(changes)
                    self.assertTrue(all(item['position'] > row['modifier_positions']['texture'] for item in changes))
                    self.assertTrue(any(item['suffix_id'] != item['surface_id'] for item in changes))
                    self.assertTrue(any(item['hidden_max_abs'] > 0 for item in changes))

    def test_surface_rejects_changed_prefix_or_modifier_hidden(self):
        with self.assertRaisesRegex(ValueError, 'modifier hidden state'):
            conditioning.diagnose(self.make_pipe(causal=False), study.sct.PROMPT,
                                  study.sct.SCENES, study.sct.TOKENS, mode='surface')
        pipe = self.make_pipe()
        encode = pipe.tokenizer.encode
        def changed_prefix(text, add_special_tokens=True):
            ids = encode(text, add_special_tokens=add_special_tokens)
            return [21 if value == 20 else value for value in ids] if 'surface' in text else ids
        pipe.tokenizer.encode = changed_prefix
        with self.assertRaisesRegex(ValueError, 'at/before Texture'):
            conditioning.diagnose(pipe, study.sct.PROMPT, study.sct.SCENES, study.sct.TOKENS, mode='surface')


if __name__ == '__main__':
    unittest.main()
