"""Historical copies must retain every byte and reject changed generation evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.wood_spoon_composition_v1 import reuse_composition as R


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'old/fullrank'
        self.source.mkdir(parents=True)
        self.destination = self.root / 'new/72/step200/fullrank'
        self.addCleanup(patch.stopall)
        patch.object(R, 'SOURCES', {200: self.source.parent}).start()
        # Windows forbids the real Linux <S*>/<T*> embedding filenames. Pair
        # tensor validation is isolated here; file locks, images and ledgers are real.
        self.check_training = R.training_checked
        patch.object(R, 'training_checked').start()
        locks, selected, refs, copies = {}, {}, {}, {}
        def lock(path):
            locks[str(path)] = R.I.sha256(path)
        for name, token, step, total, count in (
                ('subject', '<S*>', 200, 600, 72), ('texture', '<T*>', 500, 1000, 16)):
            run = self.root / ('training_' + name)
            checkpoint = run / f'checkpoint-{step}'
            checkpoint.mkdir(parents=True)
            config = {'stage': 'train', 'run': {'study': 'reuse_fixture', 'variant': 'test', 'seed': 42},
                'args': {'modifier_token': token, 'max_train_steps': total,
                         'initializer_token': 'spoon' if name == 'subject' else 'texture'}}
            R.I.write(run / 'config.yaml', config)
            R.I.write(checkpoint / 'adaptation_config.json',
                      {'modifier_tokens': [token], 'adaptation_mode': 'token_local_kv'})
            weight = 'pytorch_token_local_kv_weights.bin'
            (checkpoint / weight).write_bytes(b'fixture adapter, no tensor deserialization')
            embedding = checkpoint / (name + '_embedding.bin')
            embedding.write_bytes(b'fixture embedding')
            hashes = {}
            for path in (run / 'config.yaml', checkpoint / 'adaptation_config.json',
                         checkpoint / weight, embedding):
                lock(path)
                hashes[str(path)] = locks[str(path)]
            selected[name] = {'method': 'fullrank', 'branch': name, 'token': token,
                'status': 'verified', 'step': step, 'steps': step, 'training_steps': total,
                'weight': weight, 'concept_id': 'wood_spoon' if name == 'subject' else 'wood_spoon_texture',
                'paired_embedding_finite': True, 'run': str(run), 'checkpoint': str(checkpoint), 'hashes': hashes}
            refs[name] = []
            for number in range(count):
                image = run / f'train{number}.png'
                Image.new('RGB', (2, 2), (number, 20, 30)).save(image)
                lock(image)
                copied = self.source / f'training_references/{name}/train_{number:03d}.png'
                copied.parent.mkdir(parents=True, exist_ok=True)
                copied.write_bytes(image.read_bytes())
                copies[str(copied)] = locks[str(image)]
                refs[name].append({'image': str(image), 'caption': f'a {token}',
                                   'image_sha256': locks[str(image)]})
        audit = self.root / 'audit.json'
        R.I.write(audit, {'sources': {name: {'fullrank': item} for name, item in selected.items()},
                         'references': refs})
        lock(audit)
        R.I.write(self.source / 'training_references.json', {'references': {
            name: [{**row, 'review_image': f'training_references/{name}/train_{i:03d}.png'}
                   for i, row in enumerate(rows)] for name, rows in refs.items()}})
        rows = [{key: value for key, value in row.items() if key != 'subject_images'}
                for row in R.I.prompt_rows('fullrank', 'texture', 500, 200)]
        self.ledger = []
        for row in rows:
            path = self.source / row['image_path']
            path.parent.mkdir(exist_ok=True)
            Image.new('RGB', (512, 512), (row['seed'], 30, 40)).save(path)
            self.ledger.append({**row, 'status': 'ok', 'subject_adapter_loaded': True,
                'attribute_adapter_loaded': True, 'attribute_token': '<T*>', 'image_sha256': R.I.sha256(path)})
        self.write_lines('generation_manifest.jsonl', rows)
        self.write_lines('generation_status.jsonl', self.ledger)
        self.state = {'status': 'succeeded', 'generated_images': 31, 'expected_images': 31,
            'method': 'fullrank', 'branch': 'texture', 'subject_step': 200, 'attribute_step': 500,
            'sampling': copy.deepcopy(R.I.SAMPLING), 'prompt': R.I.PROMPTS['texture'],
            'training_images': {'subject': 72, 'texture': 16}, 'source_locks': locks,
            'reference_copy_hashes': copies, 'audit': str(audit), 'safety_filtered': 0,
            'internal_adapter_slots': {'subject': '<S*>', 'material': '<T*>'},
            'attribute_semantics': 'texture', 'subject_training': selected['subject'],
            'attribute_training': selected['texture'], 'inference_git': {'commit': 'historical', 'branch': 'old'}}
        self.save_state()

    def write_lines(self, name, rows):
        (self.source / name).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')

    def save_state(self):
        R.I.write(self.source / 'provenance.json', self.state)

    def reuse(self):
        return R.copy_completed(self.source, self.destination, 200, 'fullrank')

    def test_copy_preserves_all_original_bytes_and_old_git(self):
        before = R.tree_hashes(self.source)
        receipt = self.reuse()
        self.assertEqual(R.tree_hashes(self.source), before)
        after = R.tree_hashes(self.destination)
        after.pop('reuse_receipt.json')
        self.assertEqual(after, before)
        self.assertEqual(receipt['original_inference_git'], self.state['inference_git'])
        self.assertEqual(receipt['generated_images'], 31)

    def test_bad_sampler_or_incomplete_status_rejected_before_copy(self):
        for field, value in (('sampling', {**R.I.SAMPLING, 'guidance_scale': 7.5}), ('status', 'running')):
            with self.subTest(field=field):
                original = self.state[field]
                self.state[field] = value
                self.save_state()
                with self.assertRaises(ValueError): self.reuse()
                self.assertFalse(self.destination.exists())
                self.state[field] = original

    def test_duplicate_seed_and_false_adapter_flag_rejected(self):
        original = copy.deepcopy(self.ledger)
        for key, value in (('seed', 30), ('attribute_adapter_loaded', False)):
            self.ledger = copy.deepcopy(original)
            self.ledger[1][key] = value
            self.write_lines('generation_status.jsonl', self.ledger)
            with self.assertRaises(ValueError): self.reuse()

    def test_changed_image_and_locked_source_rejected(self):
        image = self.source / self.ledger[0]['image_path']
        old = image.read_bytes()
        Image.new('RGB', (512, 512), 'red').save(image)
        with self.assertRaises(ValueError): self.reuse()
        image.write_bytes(old)
        locked = Path(next(iter(self.state['source_locks'])))
        locked.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.reuse()

    def test_existing_destination_overlap_and_nonfrozen_source_rejected(self):
        self.destination.mkdir(parents=True)
        with self.assertRaises(ValueError): self.reuse()
        with self.assertRaises(ValueError): R.copy_completed(self.source, self.source / 'nested', 200, 'fullrank')
        with self.assertRaises(ValueError): R.copy_completed(self.root, self.root / 'other', 200, 'fullrank')

    def test_real_training_metadata_validator_rejects_wrong_training_total(self):
        audit = R.I.read_json(self.state['audit'])
        # Virtual token filenames only: this validator checks membership, never
        # deserializes tensors; the separate copy tests validate real file locks.
        for field, name, token in (('subject_training', 'subject', '<S*>'),
                                   ('attribute_training', 'texture', '<T*>')):
            item = self.state[field]
            virtual = str(Path(item['checkpoint']) / (token + '.bin'))
            item['hashes'][virtual] = 'fixture'
            self.state['source_locks'][virtual] = 'fixture'
            audit['sources'][name]['fullrank'] = copy.deepcopy(item)
        self.check_training(self.state, audit, 'fullrank', 200)
        self.state['subject_training']['training_steps'] = 1000
        audit['sources']['subject']['fullrank']['training_steps'] = 1000
        with self.assertRaises(ValueError):
            self.check_training(self.state, audit, 'fullrank', 200)

    def test_symlink_guard_rejects_before_copy(self):
        with patch.object(Path, 'is_symlink', return_value=True):
            with self.assertRaises(ValueError): self.reuse()


if __name__ == '__main__':
    unittest.main()
