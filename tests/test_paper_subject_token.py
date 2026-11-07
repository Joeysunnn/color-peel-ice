"""Protect the requested prompts, seed pairing and object-local inference evidence."""

from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.paper_subject_token_v1 import run


class SubjectTokenTests(unittest.TestCase):
    def test_exact_nine_prompts_and_inclusive_369_image_plan(self):
        prompts = {
            'a photo of <S*>',
            'a photo of <S*> in the snow',
            'a photo of <S*> in the street',
            'a photo of <S*> surrounded by flowers',
            'a photo of <S*> in blue color',
            'a photo of <S*> in white color',
            'a photo of <S*> in red color',
            'a photo of <S*> with a metal surface',
            'a photo of <S*> with a matte surface',
        }
        rows = run.plan()
        self.assertEqual(len(rows), 369)
        self.assertEqual(len({row['id'] for row in rows}), 369)
        self.assertEqual(len({row['image_path'] for row in rows}), 369)
        self.assertEqual(Counter(row['prompt'] for row in rows), dict.fromkeys(prompts, 41))
        for prompt in prompts:
            self.assertEqual({row['seed'] for row in rows if row['prompt'] == prompt}, set(range(20, 61)))
        for row in rows:
            self.assertNotIn('mailbox', row['prompt'])
            self.assertEqual(row['prompt'].count('<S*>'), 1)
            self.assertFalse(any(token in row['prompt'] for token in ('<C*>', '<M*>', '<T*>')))
            self.assertEqual(row['branch_steps'], {'subject': 1000})
            self.assertEqual(row['guidance_scale'], 3.5)
            self.assertEqual(row['negative_prompt'], '')

    def test_four_gpu_partition_keeps_all_prompts_for_each_seed_together(self):
        rows, gpus = run.plan(), [0, 1, 2, 3]
        parts = {gpu: run.shard(rows, gpus, gpu) for gpu in gpus}
        self.assertEqual([len(parts[gpu]) for gpu in gpus], [99, 90, 90, 90])
        combined = [row for part in parts.values() for row in part]
        self.assertEqual(Counter(row['id'] for row in combined), Counter(row['id'] for row in rows))
        for seed in range(20, 61):
            owners = [gpu for gpu, part in parts.items() if any(row['seed'] == seed for row in part)]
            self.assertEqual(len(owners), 1)
            self.assertEqual(sum(row['seed'] == seed for row in parts[owners[0]]), 9)

    def test_checkpoint_copy_preserves_producer_files_and_adds_copy_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / 'producer', root / 'object/ckpts/subject/step1000'
            source.mkdir()
            # A portable token filename exercises the copier on Windows, without loading a model.
            content = {'weights.bin': b'adapter', 'subject.bin': b'embedding',
                       'adaptation_config.json': b'{"rank":64}\n',
                       'receipt.json': b'{"producer":"old_git","checkpoint":"original"}\n'}
            for name, value in content.items():
                (source / name).write_bytes(value)
            item = {'checkpoint': str(source), 'path': str(source), 'weight': 'weights.bin',
                    'token': 'subject', 'hashes': {str(source / name): run.sha256(source / name) for name in content}}
            before = copy.deepcopy(item)
            selected, copies = run.copy_checkpoint(item, target)
            self.assertEqual(item, before)
            self.assertEqual(selected['checkpoint'], str(target))
            self.assertEqual(selected['path'], str(target))
            self.assertEqual(selected['original_checkpoint'], str(source))
            self.assertEqual(len(copies), 4)
            for name, value in content.items():
                self.assertEqual((source / name).read_bytes(), value)
                self.assertEqual((target / name).read_bytes(), value)
                self.assertEqual(selected['hashes'][str(target / name)], item['hashes'][str(source / name)])
            with self.assertRaises(FileExistsError):
                run.copy_checkpoint(item, target)

    def fixture(self, output, rows, *, filtered=False):
        records = []
        for row in rows:
            path = output / row['image_path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('image bytes: ' + row['id']).encode())
            gpu = (row['seed'] - 20) % 4
            records.append({**row, 'loaded_branches': ['subject'], 'loaded_source_group': f'gpu{gpu}',
                            'initial_denoiser_input': {'sha256': f'noise{row["seed"]}'},
                            'image_sha256': run.sha256(path), 'nsfw_content_detected': filtered,
                            'status': 'safety_filtered' if filtered else 'ok'})
        return records

    def write_ledger(self, output, gpu, records):
        ledger = output / f'groups/gpu{gpu}/generation_status.jsonl'
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf8')
        return ledger

    def test_collect_keeps_filtered_results_and_audits_same_seed_inputs(self):
        rows = [row for row in run.plan() if row['seed'] == 20][:2]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            records = self.fixture(output, rows, filtered=True)
            ledger = self.write_ledger(output, 0, records)
            original = ledger.read_bytes()
            generated = run.collect(output, rows, [0, 1, 2, 3])
            self.assertEqual(generated, records)
            self.assertEqual(ledger.read_bytes(), original)
            audit = json.loads((output / 'initial_input_audit.json').read_text())
            self.assertTrue(audit['same_seed_equal_across_prompts'])
            self.assertEqual(audit['by_seed'], {'20': {'sha256': 'noise20'}})
            saved = [json.loads(line) for line in (output / 'generation_status.jsonl').read_text().splitlines()]
            self.assertEqual(saved, records)

    def test_collect_rejects_altered_plan_bytes_or_gpu_without_deleting_evidence(self):
        rows = [run.plan()[0]]
        for change in ('seed', 'prompt', 'image_bytes', 'gpu'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                records = self.fixture(output, rows)
                gpu = 0
                if change == 'seed':
                    records[0]['seed'] = 21
                elif change == 'prompt':
                    records[0]['prompt'] += ' mailbox'
                elif change == 'image_bytes':
                    (output / rows[0]['image_path']).write_bytes(b'changed image')
                else:
                    gpu = 1
                    records[0]['loaded_source_group'] = 'gpu1'
                ledger = self.write_ledger(output, gpu, records)
                original = ledger.read_bytes()
                with self.assertRaisesRegex(ValueError, 'generated record differs'):
                    run.collect(output, rows, [0, 1, 2, 3])
                self.assertEqual(ledger.read_bytes(), original)
                self.assertTrue((output / rows[0]['image_path']).is_file())

    def test_collect_rejects_different_initial_noise_for_the_same_seed(self):
        rows = [row for row in run.plan() if row['seed'] == 20][:2]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            records = self.fixture(output, rows)
            records[1]['initial_denoiser_input'] = {'sha256': 'different-noise'}
            ledger = self.write_ledger(output, 0, records)
            original = ledger.read_bytes()
            with self.assertRaisesRegex(ValueError, 'same-seed initial denoiser input changed'):
                run.collect(output, rows, [0, 1, 2, 3])
            self.assertEqual(ledger.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
