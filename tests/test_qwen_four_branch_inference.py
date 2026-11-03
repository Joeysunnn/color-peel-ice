"""Fixed diagnostic protocol and same-seed sampling checks; no model downloads."""

from collections import Counter
import copy
import unittest

from experiments.sd21_qwen_initializers_v1 import inference

try:
    import torch
except ModuleNotFoundError:
    torch = None


def manifest():
    words = {'subject': 'pot', 'color': 'cyan', 'material': 'matte', 'texture': 'striped'}
    return {'subject_category': 'teapot', 'branches': {
        branch: {'token': inference.TOKENS[branch], 'initializer': word, 'initializer_token_id': index,
                 'description': 'unused free description'} for index, (branch, word) in enumerate(words.items())}}


class FourBranchPlanTests(unittest.TestCase):
    def test112_unique_rows_and_exact_control_counts(self):
        rows = inference.plan(manifest())
        self.assertEqual(len(rows), 112)
        self.assertEqual(len({row['id'] for row in rows}), 112)
        self.assertEqual(len({row['image_path'] for row in rows}), 112)
        expected = {(branch + '_only', condition): 8 for branch in inference.BRANCHES
                    for condition in ('native', 'literal', 'learned')}
        expected[('scmt', 'learned')] = 16
        self.assertEqual(Counter((row['mode'], row['condition']) for row in rows), expected)
        for row in rows:
            self.assertEqual(row['negative_prompt'], '')
            self.assertIn(row['seed'], (31, 38))
            self.assertEqual(row['method'], 'lora_r64' if row['branch_steps'] else 'stock')

    def test_native_literal_and_learned_subject_have_explicit_distinct_meaning(self):
        rows = {row['condition']: row for row in inference.plan(manifest())
                if row['mode'] == 'subject_only' and row['case'] == 'street' and row['seed'] == 31}
        self.assertEqual(rows['native']['prompt'], 'a photo of an object in the street')
        self.assertEqual(rows['literal']['prompt'], 'a photo of a teapot in the street')
        self.assertEqual(rows['learned']['prompt'], 'a photo of <S*> teapot in the street')
        self.assertEqual(rows['native']['control_role'], 'generic_object_base')
        self.assertEqual(rows['literal']['control_role'], 'subject_category_base')
        self.assertEqual(rows['learned']['branch_steps'], {'subject': 1000})
        self.assertFalse(rows['native']['branch_steps'] or rows['literal']['branch_steps'])

    def test_literal_attributes_use_selected_words_and_material_surface_grammar(self):
        endings = {'color': ' in cyan color', 'material': ' with a matte surface', 'texture': ' with striped texture'}
        for row in inference.plan(manifest()):
            if row['mode'] in ('subject_only', 'scmt'):
                continue
            branch = row['mode'].removesuffix('_only')
            self.assertIn(row['case'], inference.OBJECTS)
            prefix = f'a photo of a {row["case"]}'
            if row['condition'] == 'literal':
                self.assertEqual(row['prompt'], prefix + endings[branch])
            elif row['condition'] == 'native':
                self.assertEqual(row['prompt'], prefix)
            else:
                expected = endings[branch].replace(manifest()['branches'][branch]['initializer'], inference.TOKENS[branch])
                self.assertEqual(row['prompt'], prefix + expected)
            self.assertNotIn('orange', row['prompt'])
            self.assertNotIn('metal', row['prompt'])

    def test_composition_pairs_steps_and_all_loaded_tokens_occur_once(self):
        for row in inference.plan(manifest()):
            for branch, token in inference.TOKENS.items():
                self.assertEqual(row['prompt'].count(token), int(branch in row['branch_steps']))
            self.assertEqual(row['guidance_scale'], 6.0 if row['mode'] == 'color_only' else 3.5)
            if row['mode'] == 'scmt':
                self.assertEqual(row['branch_steps'], inference.STEPS[row['phase']])
                body = 'a photo of <S*> teapot in <C*> color with a <M*> surface with <T*> texture'
                self.assertEqual(row['prompt'], (body + ' ' + inference.SCENES[row['case']]).strip())
            elif row['branch_steps']:
                branch = row['mode'].removesuffix('_only')
                self.assertEqual(row['branch_steps'], {branch: inference.STEPS['final'][branch]})

    def test_unvalidated_literals_and_category_tokens_are_rejected(self):
        bad = manifest()
        bad['branches']['texture']['initializer'] = 'fine grain'
        with self.assertRaises(ValueError):
            inference.plan(bad)
        bad = manifest()
        bad['subject_category'] = '<S*> teapot'
        with self.assertRaises(ValueError):
            inference.plan(bad)

    def test_measured_initial_input_changes_are_not_silently_accepted(self):
        inputs = {}
        captured = {'sha256': 'same', 'shape': [2, 4, 64, 64], 'dtype': 'torch.float16', 'timestep': 999.0}
        inference.register_initial_input(inputs, 31, captured)
        inference.register_initial_input(inputs, 31, copy.deepcopy(captured))
        with self.assertRaises(ValueError):
            inference.register_initial_input(inputs, 31, {**captured, 'sha256': 'changed'})
        with self.assertRaises(ValueError):
            inference.register_initial_input(inputs, 38, {})


@unittest.skipIf(torch is None, 'requires server torch')
class SameSeedSamplingTests(unittest.TestCase):
    def test_stock_single_and_four_branch_receive_equal_fresh_noise_despite_cfg_difference(self):
        rows = inference.plan(manifest())
        selected = [next(row for row in rows if row['mode'] == mode and row['condition'] == condition
                         and row['seed'] == 31) for mode, condition in (
                             ('subject_only', 'native'), ('subject_only', 'learned'),
                             ('color_only', 'learned'), ('material_only', 'learned'),
                             ('texture_only', 'learned'), ('scmt', 'learned'))]
        samples, generators = [], []
        for row in selected:
            kwargs = inference.generation_kwargs(row, 'cpu')
            generators.append(kwargs['generator'])
            samples.append(torch.randn((1, 4, 64, 64), generator=kwargs['generator'], dtype=torch.float32).half())
            self.assertEqual(kwargs['num_inference_steps'], 100)
            self.assertEqual((kwargs['width'], kwargs['height']), (512, 512))
            self.assertGreater(kwargs['guidance_scale'], 1)
        self.assertEqual(len({id(generator) for generator in generators}), len(selected))
        for sample in samples[1:]:
            torch.testing.assert_close(sample, samples[0], rtol=0, atol=0)
        other = inference.generation_kwargs({**selected[0], 'seed': 38}, 'cpu')['generator']
        self.assertFalse(torch.equal(samples[0], torch.randn((1, 4, 64, 64), generator=other).half()))


if __name__ == '__main__':
    unittest.main()
