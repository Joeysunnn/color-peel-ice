"""Prevent CFG comparisons from changing prompts, checkpoints or seeds."""

from collections import Counter
from pathlib import Path
import tempfile
import unittest

from experiments.sd21_qwen_cfg_compare_v1 import run


def manifest():
    words = {'subject': 'mailbox', 'color': 'orange', 'material': 'painted', 'texture': 'uniform'}
    return {'subject_category': 'mailbox', 'branches': {
        b: {'token': run.contract.TOKENS[b], 'initializer': word} for b, word in words.items()}}


class CFGCompareTests(unittest.TestCase):
    def test297_independent_branch_rows_and_exact_checkpoint_steps(self):
        rows = run.plan(manifest())
        self.assertEqual(len(rows), 297)
        self.assertEqual(len({r['id'] for r in rows}), 297)
        self.assertEqual(len({r['image_path'] for r in rows}), 297)
        self.assertEqual(Counter(r['mode'] for r in rows), {'subject_only': 165, 'material_only': 132})
        for row in rows:
            branch = row['mode'].removesuffix('_only')
            self.assertEqual(row['branch_steps'], {branch: 1000 if branch == 'subject' else 150})
            self.assertEqual(row['negative_prompt'], '')

    def test_each_prompt_seed_triplet_changes_only_cfg_and_identity_paths(self):
        groups = {}
        for row in run.plan(manifest()):
            groups.setdefault((row['mode'], row['case'], row['seed']), []).append(row)
        self.assertEqual(len(groups), 99)
        for rows in groups.values():
            self.assertEqual({r['guidance_scale'] for r in rows}, {3.5, 5.0, 7.5})
            comparable = [{k: v for k, v in r.items() if k not in ('guidance_scale', 'id', 'image_path')} for r in rows]
            self.assertTrue(all(r == comparable[0] for r in comparable))
        for mode, case, _ in groups:
            self.assertEqual({seed for m, c, seed in groups if (m, c) == (mode, case)}, set(range(30, 41)))

    def test_wrong_step_cannot_be_silently_loaded(self):
        item = {'branch': 'material', 'step': 200, 'steps': 200}
        with self.assertRaisesRegex(ValueError, 'wrong branch, step'):
            run.validate_selection(item, 'material', manifest(), {}, {})

    def test_gallery_has_adjacent_cfg_columns_and_correct_branch_title(self):
        rows = [r for r in run.plan(manifest()) if r['mode'] == 'material_only']
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            run.gallery(output, rows, [], {'status': 'prepared'}, 'material_only')
            page = (output / 'groups/material_only/index.html').read_text(encoding='utf8')
            self.assertIn('material150', page)
            self.assertIn('<th>CFG3.5</th><th>CFG5</th><th>CFG7.5</th>', page)


if __name__ == '__main__':
    unittest.main()
