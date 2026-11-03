"""CPU-only protocol checks; no models, assets, CUDA or inference are loaded."""

from collections import Counter
import unittest

from experiments.sd21_tokenverse_isolation_v1.inference import COMBINATIONS, plan


class TokenIsolationPlanTests(unittest.TestCase):
    def test_exact112_unique_outputs_and_fixed_group_counts(self):
        rows = plan()
        self.assertEqual(len(rows), 112)
        self.assertEqual(len({row['id'] for row in rows}), 112)
        self.assertEqual(len({row['image_path'] for row in rows}), 112)
        self.assertEqual(Counter((row['phase'], row['mode']) for row in rows),
            {('final', 'sct'): 64, ('mid', 'sct'): 16, ('final', 'subject_only'): 16,
             ('final', 'texture_only'): 12, ('final', 'color_only'): 4})
        self.assertEqual({row['method'] for row in rows}, {'lora_r64'})
        self.assertEqual({row['seed'] for row in rows}, {31, 38})
        self.assertTrue(all(row['negative_prompt'] == '' for row in rows))

    def test_factorial_order_arm_mapping_and_matched_saved_steps(self):
        self.assertEqual(COMBINATIONS, ('ccc', 'icc', 'cic', 'cci', 'iic', 'ici', 'cii', 'iii'))
        for phase, combos, steps in (
                ('final', COMBINATIONS, {'subject': 1000, 'color': 150, 'texture': 1000}),
                ('mid', ('ccc', 'iii'), {'subject': 500, 'color': 100, 'texture': 500})):
            rows = [row for row in plan() if (row['phase'], row['mode']) == (phase, 'sct')]
            self.assertEqual(list(dict.fromkeys(row['condition'] for row in rows)), list(combos))
            for combo in combos:
                selected = [row for row in rows if row['condition'] == combo]
                self.assertEqual({(row['scene'], row['seed']) for row in selected},
                    {(scene, seed) for scene in ('street', 'snow', 'flowers', 'beach') for seed in (31, 38)})
                for row in selected:
                    self.assertEqual(row['branch_steps'], steps)
                    self.assertEqual(list(row['branch_arms']), ['subject', 'color', 'texture'])
                    self.assertEqual(list(row['branch_arms'].values()),
                                     [{'c': 'control', 'i': 'isolation'}[letter] for letter in combo])

    def test_single_branch_anchors_do_not_load_or_prompt_other_learned_concepts(self):
        expected = {
            'subject': (1000, {(scene, seed) for scene in ('street', 'snow', 'flowers', 'beach') for seed in (31, 38)}),
            'texture': (1000, {('mug', 31), ('mug', 38), ('vase', 31), ('vase', 38), ('chair', 38), ('sofa', 31)}),
            'color': (150, {('mug', 31), ('vase', 38)})}
        for branch, (step, anchors) in expected.items():
            for arm in ('control', 'isolation'):
                rows = [row for row in plan() if row['mode'] == branch + '_only' and row['condition'] == arm]
                self.assertEqual({(row['scene'], row['seed']) for row in rows}, anchors)
                self.assertEqual(len(rows), len(anchors))
                for row in rows:
                    self.assertEqual(row['branch_arms'], {branch: arm})
                    self.assertEqual(row['branch_steps'], {branch: step})

    def test_prompts_are_exact_and_each_loaded_token_occurs_once(self):
        suffixes = {'street': 'in the street', 'snow': 'in the snow',
                    'flowers': 'surrounded by flowers', 'beach': 'at the beach'}
        for row in plan():
            if row['mode'] == 'sct':
                expected = 'a photo of <S*> mailbox in <C*> color with <T*> texture ' + suffixes[row['scene']]
            elif row['mode'] == 'subject_only':
                expected = 'a photo of <S*> mailbox ' + suffixes[row['scene']]
            elif row['mode'] == 'texture_only':
                expected = f'a photo of a {row["scene"]} with <T*> texture'
            else:
                expected = f'a photo of a {row["scene"]} in <C*> color'
            self.assertEqual(row['prompt'], expected)
            for branch, token in (('subject', '<S*>'), ('color', '<C*>'), ('texture', '<T*>')):
                self.assertEqual(row['prompt'].count(token), int(branch in row['branch_arms']))
            self.assertNotIn('<M*>', row['prompt'])


if __name__ == '__main__':
    unittest.main()
