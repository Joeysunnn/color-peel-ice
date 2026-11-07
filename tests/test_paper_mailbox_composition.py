"""Guard the requested composition and exclude obsolete Texture exports."""

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.paper_mailbox_composition_v1 import run


class CompositionTests(unittest.TestCase):
    def test_exact_prompts_seeds_and_all_four_branches(self):
        prompts = [
            'a photo of <S*> in <C*> color with a <M*> surface with <T*> texture',
            'a photo of <S*> in <C*> color with a <M*> surface with <T*> texture in the street',
            'a photo of <S*> in <C*> color with a <M*> surface with <T*> texture in the snow',
            'a photo of <S*> in <C*> color with a <M*> surface with <T*> texture surrounded by flowers',
        ]
        rows = run.plan()
        self.assertEqual(list(dict.fromkeys(row['prompt'] for row in rows)), prompts)
        self.assertEqual(len(rows), 164)
        self.assertEqual(len({row['id'] for row in rows}), 164)
        self.assertEqual(len({row['image_path'] for row in rows}), 164)
        for prompt in prompts:
            self.assertEqual([row['seed'] for row in rows if row['prompt'] == prompt], list(range(20, 61)))
        for row in rows:
            self.assertEqual(row['branch_steps'], {'subject': 1000, 'color': 150, 'material': 150, 'texture': 300})
            self.assertEqual(row['mode'], 'scmt')
            self.assertEqual(row['guidance_scale'], 3.5)
            self.assertEqual(row['negative_prompt'], '')
            for token in ('<S*>', '<C*>', '<M*>', '<T*>'):
                self.assertEqual(row['prompt'].count(token), 1)

    def test_reject_wrong_step_wrong_texture_source_or_unpaired_file(self):
        directory = run.checkpoint_paths(Path.cwd())['texture']
        token, weight = '<T*>', 'pytorch_lora_kv_weights.bin'
        item = {'branch': 'texture', 'step': 300, 'steps': 300, 'method': 'lora_r64',
                'token': token, 'embedding_dim': 1024, 'base_model': {'model': 'SD2.1'},
                'status': 'verified', 'paired_embedding_finite': True, 'weight': weight,
                'checkpoint': str(directory),
                'hashes': {str(directory / name): name for name in (weight, token + '.bin')}}
        with patch.object(run, 'read_json', return_value=copy.deepcopy(item)):
            run.validate_export(item, 'texture', directory, item['base_model'])
            changes = [{'step': 500}, {'token': '<M*>'}, {'branch': 'material'},
                       {'base_model': {'model': 'SD1.5'}}, {'paired_embedding_finite': False},
                       {'checkpoint': str(directory).replace('texture_legacy_t16__ad3a393', 'mt_vlm__16c1e7e')},
                       {'hashes': {str(directory / weight): weight}},
                       {'hashes': {key: 'changed' for key in item['hashes']}}]
            for change in changes:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    run.validate_export({**item, **change}, 'texture', directory, item['base_model'])

    def test_subject_relocation_keeps_original_receipt_hashes(self):
        directory = Path.cwd() / 'object/ckpts/subject/cb169c6/step1000'
        original = Path.cwd() / 'original/subject/ckpts/step1000'
        item = {'branch': 'subject', 'step': 1000, 'steps': 1000, 'method': 'lora_r64',
                'token': '<S*>', 'embedding_dim': 1024, 'base_model': {'model': 'SD2.1'},
                'status': 'verified', 'paired_embedding_finite': True,
                'weight': 'pytorch_lora_kv_weights.bin', 'checkpoint': str(original),
                'hashes': {str(original / name): name for name in ('pytorch_lora_kv_weights.bin', '<S*>.bin')}}
        selected = {**item, 'checkpoint': str(directory), 'original_checkpoint': str(original),
                    'hashes': {**item['hashes'], **{str(directory / Path(key).name): value
                              for key, value in item['hashes'].items()}}}
        with patch.object(run, 'read_json', return_value=item):
            run.validate_export(selected, 'subject', directory, item['base_model'])
            selected['hashes'].pop(str(original / '<S*>.bin'))
            with self.assertRaisesRegex(ValueError, 'producer hashes'):
                run.validate_export(selected, 'subject', directory, item['base_model'])

    def test_uses_existing_scmt_engine(self):
        from experiments.sd21_qwen_midpoint_transfer_v1.run import run_group, collect
        self.assertIs(run.run_group, run_group)
        self.assertIs(run.collect, collect)

    def test_failed_generation_remains_failed_if_ledger_collection_also_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            audit = {'inference_git': {'commit': 'test', 'branch': 'codex/test'}}
            with patch.dict(os.environ), patch.object(run, 'require_idle'), \
                    patch.object(run, 'prepare', return_value=(output, run.plan(), audit)), \
                    patch.object(run, 'run_group', side_effect=RuntimeError('generation failed')), \
                    patch.object(run, 'collect', side_effect=ValueError('broken ledger')):
                with self.assertRaisesRegex(RuntimeError, '^generation failed$'):
                    run.run(output, 0)
            state = run.read_json(output / 'status.json')
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['error'], 'RuntimeError: generation failed')
            self.assertEqual(state['collection_error'], 'ValueError: broken ledger')


if __name__ == '__main__':
    unittest.main()
