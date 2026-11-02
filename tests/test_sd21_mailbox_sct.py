"""CPU source-routing checks for the requested three-branch experiment."""
import os
import unittest
from unittest.mock import call, patch

from experiments.sd21_mailbox_sct_v1 import run as sct


class SCTPlanTest(unittest.TestCase):
    def test_three_learned_tokens_and_matched_seeds(self):
        rows = [r for m in sct.METHODS for r in sct.prompt_rows(m)]
        self.assertEqual(len(rows), 22)
        self.assertEqual(len({r['id'] for r in rows}), 22)
        self.assertEqual({tuple(r['steps'].items()) for r in rows},
                         {(('subject', 1000), ('color', 150), ('texture', 1000))})
        for m in sct.METHODS:
            self.assertEqual([r['seed'] for r in rows if r['method'] == m], list(range(30, 41)))
        for token in ('<S*>', '<C*>', '<T*>'):
            self.assertEqual(sct.PROMPT.count(token), 1)
        self.assertNotIn('orange', sct.PROMPT)
        with self.assertRaises(ValueError):
            sct.prompt_rows('mixed')

    def test_color150_added_to_existing_s1000_t1000_audit(self):
        revision, model = {'commit': 'new', 'branch': 'new'}, {'id': 'sd21'}
        audit = {'sources': {'subject': {m: {'base_model': model} for m in sct.METHODS}},
                 'references': {}, 'training_gits': {}, 'source_locks': {}, 'inference_git': revision}
        with patch.dict(os.environ), patch.object(sct.st, 'prepare', return_value=audit) as old, \
             patch.object(sct.color_contract, 'verify_dataset', return_value={'rows': 9, 'references': [], 'source_locks': {}}), \
             patch.object(sct.st.source, '_pair', return_value=({'token': '<C*>'}, {})) as pair, \
             patch.object(sct.st, 'git_info', return_value=revision):
            result = sct.prepare()
            self.assertEqual(os.environ[sct.color_contract.DATASET_ENV], str(sct.COLOR_ROOT / 'dataset'))
        old.assert_called_once_with(1000, 1000)
        self.assertEqual(pair.call_args_list, [call(sct.COLOR_ROOT, 'color', m, 150, sct.COLOR_GIT,
            model, recipe_contract=sct.color_contract) for m in sct.METHODS])
        self.assertEqual(result['training_images'], {'subject': 11, 'color': 9, 'texture': 16})


if __name__ == '__main__':
    unittest.main()
