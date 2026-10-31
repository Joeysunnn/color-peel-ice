"""CPU tests for exact historical controls, without sampling or tensor loading."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.wood_spoon_composition_v1 import control_inference as C
from experiments.wood_spoon_composition_v1 import control_reuse as R


class ControlReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        payload = self.root / 'paired_adapter_embedding.fixture'
        payload.write_bytes(b'unchanged selected tensor fixture')
        self.locks = {str(payload): R.I.sha256(payload)}
        selection = {'method': 'lora_r64', 'concept_id': 'wood_spoon_texture', 'steps': 500,
                     'hashes': copy.deepcopy(self.locks)}
        self.audit = {'sources': {'texture': {'lora_r64': selection},
            'subject': {'lora_r64': {**selection, 'concept_id': 'wood_spoon', 'steps': 600}}},
            'source_locks': copy.deepcopy(self.locks)}
        self.sources = {case: self.root / case for case in R.SOURCES}
        patcher = patch.object(R, 'SOURCES', self.sources)
        patcher.start()
        self.addCleanup(patcher.stop)
        for case, source in self.sources.items():
            source.mkdir()
            if case.startswith('mug_'):
                self.make_mug(case, source)
            else:
                self.make_composition(case, source)

    def records(self, source, manifest, ledger):
        for name, rows in (('generation_manifest.jsonl', manifest), ('generation_status.jsonl', ledger)):
            (source / name).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')

    def image(self, source, row):
        image = source / row['image_path']
        image.parent.mkdir(exist_ok=True)
        Image.new('RGB', (512, 512), (42, 100, 150)).save(image)
        return R.I.sha256(image)

    def make_composition(self, case, source):
        use_subject = C.CASES[case]['use_subject']
        rows = R.I.prompt_rows('lora_r64', 'texture', 500, 600, literal_subject=not use_subject)
        if use_subject:
            rows = [{k: v for k, v in row.items() if k != 'subject_images'} for row in rows]
        ledger = []
        for row in rows:
            digest = self.image(source, row) if row['seed'] == 42 else 'unselected-image-not-read'
            ledger.append({**row, 'status': 'ok', 'subject_adapter_loaded': use_subject,
                           'attribute_adapter_loaded': True, 'attribute_token': '<T*>', 'image_sha256': digest})
        self.records(source, rows, ledger)
        ref = source / 'reference.fixture'
        ref.write_bytes(b'actual reference copy')
        state = {'status': 'succeeded', 'expected_images': 31, 'generated_images': 31,
            'method': 'lora_r64', 'branch': 'texture', 'attribute_step': 500,
            'sampling': copy.deepcopy(R.I.SAMPLING), 'prompt': C.CASES[case]['prompt'],
            'attribute_training': self.audit['sources']['texture']['lora_r64'],
            'subject_training': self.audit['sources']['subject']['lora_r64'] if use_subject else None,
            'subject_step': 600 if use_subject else None,
            'training_images': {'subject': 72, 'texture': 16} if use_subject else {'texture': 16},
            'source_locks': self.locks, 'reference_copy_hashes': {str(ref): R.I.sha256(ref)},
            'inference_git': {'commit': 'historical-composition', 'branch': 'old'}}
        R.I.write(source / 'provenance.json', state)

    def make_mug(self, case, source):
        training = self.root / ('training_' + case)
        selected = self.audit['sources']['texture']['lora_r64']
        locks = dict(self.locks)
        for name, data in (('status.json', {'status': 'execution_verified'}), ('protocol.json', {}),
            ('training/training_index.json', {'runs': []}), ('step500/training_index.json', {'runs': [selected]})):
            path = training / name
            path.parent.mkdir(parents=True, exist_ok=True)
            R.I.write(path, data)
            locks[str(path)] = R.I.sha256(path)
        conditions = ('base', 'learned_texture')
        rows = [{'id': f'old_{i}', 'method': 'lora_r64', 'branch': 'texture', 'concept_id': 'wood_spoon_texture',
            'context': 'mug' if i < 2 else 'chair', 'seed': 42,
            'condition': conditions[i] if i < 2 else 'literal',
            'prompt': C.CASES['mug_with_t' if i == 1 else 'mug_without_t']['prompt'],
            'image_path': f'images/old_{i}.png'} for i in range(288)]
        ledger = [{**row, 'status': 'ok', 'image_sha256': self.image(source, row) if i < 2 else 'unused',
            'texture_adapter_loaded': i == 1, 'material_adapter_loaded': False} for i, row in enumerate(rows)]
        self.records(source, rows, ledger)
        R.I.write(source / 'summary.json', {'status': 'succeeded', 'generated_images': 288, 'missing_images': 0})
        R.I.write(source / 'provenance.json', {'status': 'succeeded', 'training_steps': 500,
            'probe_suite': 'ordinary_object_transfer', 'training': [selected], 'source_run': str(training),
            'source_locks': locks, 'git': {'commit': 'historical-native', 'branch': 'old'},
            'sampling': {'scheduler': 'PNDM', 'num_inference_steps': 100, 'guidance_scale': 3.5,
                'resolution': 512, 'dtype': 'float16', 'safety_checker': True, 'seeds': [42, 43]}})

    def run_case(self, case):
        return R.copy_existing(self.root / 'new' / case, case, self.audit)

    def test_all_four_copy_exact_png_and_preserve_original_evidence(self):
        for case, source in self.sources.items():
            with self.subTest(case=case):
                before = {str(p): R.I.sha256(p) for p in source.rglob('*') if p.is_file()}
                receipt = self.run_case(case)
                output = self.root / 'new' / case
                row = R.lines(output / 'generation_status.jsonl')[0]
                self.assertEqual(receipt['status'], 'succeeded')
                self.assertTrue(receipt['copied'])
                self.assertEqual(receipt['generated_images'], 1)
                self.assertEqual(row['prompt'], C.CASES[case]['prompt'])
                self.assertEqual(row['seed'], 42)
                self.assertEqual(row['image_sha256'], R.I.sha256(output / row['image_path']))
                self.assertEqual(before, {str(p): R.I.sha256(p) for p in source.rglob('*') if p.is_file()})
                self.assertEqual((output / 'reuse_sources/provenance.json').read_bytes(), (source / 'provenance.json').read_bytes())

    def test_other_four_return_none_without_directory(self):
        for case in set(C.CASES) - set(self.sources):
            self.assertIsNone(self.run_case(case))
            self.assertFalse((self.root / 'new' / case).exists())

    def test_changed_hash_seed_or_flags_is_rejected_without_fallback(self):
        source = self.sources['wood_spoon_with_t']
        original = (source / 'generation_status.jsonl').read_bytes()
        for key, value in (('seed', 43), ('attribute_adapter_loaded', False), ('image_sha256', 'wrong')):
            with self.subTest(key=key):
                rows = R.lines(source / 'generation_status.jsonl')
                rows[12][key] = value  # index12 is seed42
                (source / 'generation_status.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
                with self.assertRaises(ValueError): self.run_case('wood_spoon_with_t')
                self.assertFalse((self.root / 'new/wood_spoon_with_t').exists())
                (source / 'generation_status.jsonl').write_bytes(original)

    def test_changed_selection_or_sampler_is_rejected(self):
        source = self.sources['subject_with_t']
        original = R.I.read_json(source / 'provenance.json')
        for key, value in (('attribute_training', {}), ('sampling', {**R.I.SAMPLING, 'guidance_scale': 9})):
            changed = {**original, key: value}
            R.I.write(source / 'provenance.json', changed)
            with self.assertRaises(ValueError): self.run_case('subject_with_t')
        R.I.write(source / 'provenance.json', original)
        self.run_case('subject_with_t')
        with self.assertRaises(ValueError): self.run_case('subject_with_t')

    def test_mug_wrong_training_index_or_completion_rejected(self):
        source = self.sources['mug_with_t']
        state = R.I.read_json(source / 'provenance.json')
        path = Path(state['source_run']) / 'step500/training_index.json'
        R.I.write(path, {'runs': []})
        with self.assertRaises(ValueError): self.run_case('mug_with_t')
        R.I.write(self.sources['mug_without_t'] / 'summary.json', {'status': 'failed'})
        with self.assertRaises(ValueError): self.run_case('mug_without_t')


if __name__ == '__main__':
    unittest.main()
