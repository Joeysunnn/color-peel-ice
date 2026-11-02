"""CPU-only protocol and same-position context-substitution checks."""

import unittest

from experiments.sd21_context_routing_v1.conditioning import replace_sc_ids
from experiments.sd21_context_routing_v1.run import CONDITIONS, plan, require_baselines


class ContextRoutingProtocolTests(unittest.TestCase):
    def test_exact18_paired_cases_and_true_sc_semantics(self):
        rows = plan()
        self.assertEqual(len(rows), 18)
        self.assertEqual(len({row['id'] for row in rows}), 18)
        self.assertEqual(len({row['image_path'] for row in rows}), 18)
        self.assertEqual([row['condition'] for row in rows[:6]], ['baseline_sct'] * 6)
        for condition in CONDITIONS:
            selected = [row for row in rows if row['condition'] == condition]
            self.assertEqual({(row['scene'], row['seed']) for row in selected},
                             {(scene, seed) for scene in ('street', 'snow', 'flowers') for seed in (31, 38)})
            for row in selected:
                self.assertEqual(row['steps'], {'subject': 1000, 'color': 150, 'texture': 400})
                self.assertEqual(row['method'], 'lora_r64')
                self.assertEqual(row['prompt'].count('<S*>'), 1)
                self.assertEqual(row['prompt'].count('<C*>'), 1)
                self.assertEqual(row['prompt'].count('<T*>'), int(condition != 'true_SC'))
                baseline = next(case for case in rows if case['condition'] == 'baseline_sct'
                                and (case['scene'], case['seed']) == (row['scene'], row['seed']))
                expected = baseline['prompt'].replace(' with <T*> texture', '') if condition == 'true_SC' else baseline['prompt']
                self.assertEqual(row['prompt'], expected)

    def test_initializer_substitution_changes_only_two_ids(self):
        ids = list(range(77))
        positions = {'subject': 4, 'color': 7, 'texture': 10}
        auxiliary = replace_sc_ids(ids, positions, {'subject': [100], 'color': [200]})
        self.assertEqual(len(auxiliary), 77)
        self.assertEqual([i for i in range(77) if ids[i] != auxiliary[i]], [4, 7])
        self.assertEqual(auxiliary[10], ids[10])
        self.assertEqual(ids, list(range(77)))
        with self.assertRaises(ValueError):
            replace_sc_ids(ids, positions, {'subject': [100, 101], 'color': [200]})
        with self.assertRaises(ValueError):
            replace_sc_ids(ids, {**positions, 'texture': 4}, {'subject': [100], 'color': [200]})

    def test_controls_block_until_all_six_exact_baselines(self):
        baseline = [{**row, 'baseline_sha_equal': True, 'baseline_status_equal': True}
                    for row in plan() if row['condition'] == 'baseline_sct']
        require_baselines(baseline)
        for changed in (baseline[:-1], baseline[:-1] + [baseline[0]],
                        [*baseline[:-1], {**baseline[-1], 'baseline_sha_equal': False}],
                        [*baseline[:-1], {**baseline[-1], 'baseline_status_equal': False}]):
            with self.assertRaises(ValueError):
                require_baselines(changed)


if __name__ == '__main__':
    unittest.main()
