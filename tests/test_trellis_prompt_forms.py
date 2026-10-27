"""CPU checks for the exact user-requested prompts and balanced probe matrix."""

from collections import Counter
import copy
import itertools
import unittest

from experiments.subject_multicolor_multimaterial_b import inference as baseline
from experiments.subject_trellis_dataset_v1 import prompt_forms


class TrellisPromptFormsTest(unittest.TestCase):
    def test_exact_fifteen_literal_prompts(self):
        expected = {
            'a wooden <S*>', 'a <S*> made entirely of wood', 'a <S*> with a wood surface',
            'a metal <S*>', 'a <S*> made entirely of metal', 'a <S*> with a metal surface',
            'a matte <S*>', 'a <S*> made entirely of matte', 'a <S*> with a matte surface',
            'a plastic <S*>', 'a <S*> made entirely of plastic', 'a <S*> with a plastic surface',
            'a ceramic <S*>', 'a <S*> made entirely of ceramic', 'a <S*> with a ceramic surface',
        }
        records = prompt_forms.prompt_rows(500)
        self.assertEqual({row['prompt'] for row in records}, expected)
        self.assertEqual({row['prompt'] for row in prompt_forms.prompt_rows(500, (44, 45, 46, 47, 48))}, expected)
        self.assertEqual(prompt_forms.MATERIALS, ('wood', 'metal', 'matte', 'plastic', 'ceramic'))
        self.assertEqual(prompt_forms.FORMS, ('adjective', 'entirely', 'surface'))
        self.assertEqual(len(prompt_forms.LABELS), 15)
        self.assertEqual(len(set(prompt_forms.LABELS)), 15)
        self.assertEqual({row['condition'] for row in records}, set(prompt_forms.LABELS))
        for row in records:
            self.assertEqual(row['prompt'].count('<S*>'), 1)
            self.assertNotIn('photo', row['prompt'])
            self.assertNotIn('mailbox', row['prompt'])
            self.assertEqual(row['condition'], row['material'] + ' / ' + row['form'])

    def test_each_checkpoint_has_sixty_unique_balanced_rows(self):
        self.assertIs(prompt_forms.METHODS, baseline.METHODS)
        self.assertIs(prompt_forms.STEPS, baseline.STEPS)
        self.assertIs(prompt_forms.SAMPLING, baseline.SAMPLING)
        total = []
        for step in (500, 750, 1000):
            records = prompt_forms.prompt_rows(step)
            total.extend(records)
            self.assertEqual(len(records), 60)
            self.assertEqual(len({row['id'] for row in records}), 60)
            self.assertEqual(len({row['image_path'] for row in records}), 60)
            self.assertEqual({row['step'] for row in records}, {step})
            actual = Counter((row['method'], row['seed'], row['material'], row['form']) for row in records)
            expected = Counter(itertools.product(('fullrank', 'lora_r64'), (42, 43),
                ('wood', 'metal', 'matte', 'plastic', 'ceramic'), ('adjective', 'entirely', 'surface')))
            self.assertEqual(actual, expected)
            for label in prompt_forms.LABELS:
                self.assertEqual(len({row['prompt'] for row in records if row['condition'] == label}), 1)
        self.assertEqual(len(total), 180)

    def test_append_seeds_form_complete_disjoint_matrix_without_mutating_defaults(self):
        sampling_before = copy.deepcopy(baseline.SAMPLING)
        seeds = [44, 45, 46, 47, 48]
        literals = {row['condition']: row['prompt'] for row in prompt_forms.prompt_rows(500)}
        total = []
        for step in (500, 750, 1000):
            original = prompt_forms.prompt_rows(step)
            appended = prompt_forms.prompt_rows(step, tuple(seeds))
            self.assertEqual(appended, prompt_forms.prompt_rows(step, seeds))
            self.assertEqual(len(appended), 150)
            self.assertEqual(len({row['id'] for row in appended}), 150)
            self.assertEqual(len({row['image_path'] for row in appended}), 150)
            self.assertTrue({row['id'] for row in original}.isdisjoint(row['id'] for row in appended))
            self.assertTrue({row['image_path'] for row in original}.isdisjoint(row['image_path'] for row in appended))
            actual = Counter((row['method'], row['seed'], row['material'], row['form']) for row in appended)
            expected = Counter(itertools.product(('fullrank', 'lora_r64'), (44, 45, 46, 47, 48),
                ('wood', 'metal', 'matte', 'plastic', 'ceramic'), ('adjective', 'entirely', 'surface')))
            self.assertEqual(actual, expected)
            self.assertEqual({row['step'] for row in appended}, {step})
            for row in appended:
                self.assertEqual(row['prompt'], literals[row['condition']])
            self.assertEqual(prompt_forms.prompt_rows(step), original)
            total.extend(appended)
        self.assertEqual(len(total), 450)
        self.assertEqual(baseline.SAMPLING, sampling_before)
        self.assertEqual(baseline.SAMPLING['seeds'], [42, 43])
        self.assertEqual(seeds, [44, 45, 46, 47, 48])

    def test_invalid_seed_collections_are_rejected_and_zero_is_valid(self):
        bad_seeds = ((), [], 44, True, '44', {44, 45}, {}, (44, 44), [44, 44],
                     (-1,), (True,), (44, False), (44.0,), ('44',), (44, None), [44, []])
        for seeds in bad_seeds:
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                prompt_forms.prompt_rows(500, seeds)
        self.assertEqual({row['seed'] for row in prompt_forms.prompt_rows(500, (0,))}, {0})
        self.assertEqual(baseline.SAMPLING['seeds'], [42, 43])

    def test_undeclared_or_noninteger_checkpoint_is_rejected(self):
        for step in (0, 499, 501, 1001, True, 500.0, '500', None):
            with self.subTest(step=step), self.assertRaises(ValueError):
                prompt_forms.prompt_rows(step)


if __name__ == '__main__':
    unittest.main()
