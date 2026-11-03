"""Prevent checkpoint, prompt, and seed mixups in the requested transfer matrix."""

from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.sd21_qwen_midpoint_transfer_v1 import protocol as p
from experiments.sd21_qwen_midpoint_transfer_v1 import run


def manifest():
    words = {'subject': 'mailbox', 'color': 'orange', 'material': 'painted', 'texture': 'uniform'}
    return {'subject_category': 'mailbox', 'branches': {
        b: {'token': p.TOKENS[b], 'initializer': word} for b, word in words.items()}}


class MidpointTransferTests(unittest.TestCase):
    def test_requested_cases_seeds_and_counts(self):
        rows = p.plan(manifest())
        self.assertEqual(len(rows), 242)
        self.assertEqual(len({r['id'] for r in rows}), 242)
        self.assertEqual(len({r['image_path'] for r in rows}), 242)
        cases = {'subject_only': {'white', 'red', 'snow', 'street', 'flowers'},
                 'material_only': {'chair', 'vase', 'sofa', 'mug'},
                 'texture_only': {'hair', 'vase', 'sofa', 'mug'},
                 'color_only': {'bowl', 'vase', 'parrot', 'sofa', 'rose'},
                 'scmt': {'plain', 'street', 'snow', 'flowers'}}
        self.assertEqual(Counter(r['mode'] for r in rows), {m: len(c) * 11 for m, c in cases.items()})
        for mode, expected in cases.items():
            self.assertEqual({r['case'] for r in rows if r['mode'] == mode}, expected)
            for case in expected:
                self.assertEqual({r['seed'] for r in rows if (r['mode'], r['case']) == (mode, case)},
                                 set(range(30, 41)))

    def test_exact_midpoint_steps_and_disjoint_single_branch_prompts(self):
        expected = {'subject': 500, 'color': 100, 'material': 500, 'texture': 500}
        for row in p.plan(manifest()):
            branches = p.BRANCHES if row['mode'] == 'scmt' else (row['mode'].removesuffix('_only'),)
            self.assertEqual(row['branch_steps'], {b: expected[b] for b in branches})
            for branch, token in p.TOKENS.items():
                self.assertEqual(row['prompt'].count(token), int(branch in branches))
            self.assertEqual(row['guidance_scale'], 6.0 if row['mode'] == 'color_only' else 3.5)
            self.assertEqual(row['negative_prompt'], '')

    def test_hair_is_not_silently_replaced_with_chair(self):
        rows = p.plan(manifest())
        self.assertEqual({r['prompt'] for r in rows if r['mode'] == 'texture_only' and r['case'] == 'hair'},
                         {'a photo of hair with <T*> texture'})

    def test_color_prompts_are_the_five_unchanged_historical_strings(self):
        self.assertEqual({r['prompt'] for r in p.plan(manifest()) if r['mode'] == 'color_only'}, {
            'a <C*> bowl on the table', 'a <C*> vase on the shelf', 'a <C*> parrot perched on a tree',
            'a <C*> sofa in living room', 'a <C*> rose blooming in a wooden pot'})

    def test_no_unrelated_source_manifest(self):
        bad = manifest()
        bad['subject_category'] = 'teapot'
        with self.assertRaises(ValueError):
            p.plan(bad)

    def test_collect_rejects_changed_pixels_and_cross_worker_noise(self):
        rows = [next(r for r in p.plan(manifest()) if r['mode'] == mode and r['seed'] == 30)
                for mode in ('subject_only', 'color_only')]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            records = []
            for row in rows:
                image = output / row['image_path']
                image.parent.mkdir(parents=True)
                image.write_bytes(b'fixture-image')
                record = {**row, 'image_sha256': run.sha256(image), 'nsfw_content_detected': False,
                          'status': 'ok', 'initial_denoiser_input': {'sha256': 'same', 'shape': [2, 4, 64, 64]}}
                records.append(record)
                folder = output / 'groups' / row['mode']
                folder.mkdir(parents=True)
                (folder / 'generation_status.jsonl').write_text(json.dumps(record) + '\n', encoding='utf8')
            self.assertEqual(len(run.collect(output, rows)), 2)
            color_ledger = output / 'groups/color_only/generation_status.jsonl'
            bad = copy.deepcopy(records[1])
            bad['initial_denoiser_input']['sha256'] = 'different'
            color_ledger.write_text(json.dumps(bad) + '\n', encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'same-seed'):
                run.collect(output, rows)
            color_ledger.write_text(json.dumps(records[1]) + '\n', encoding='utf8')
            (output / rows[0]['image_path']).write_bytes(b'changed-image')
            with self.assertRaisesRegex(ValueError, 'saved bytes'):
                run.collect(output, rows)


if __name__ == '__main__':
    unittest.main()
