"""Check the approved center-only data, paired transfer plan and SDXL isolation."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from experiments.sd21_mailbox_center11_v1 import contract, inference, run
from experiments.subject_trellis_dataset_v1.inference import blue_green_prompt_rows


@contextmanager
def staged_fixture():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        source, output = root / 'source', root / 'new' / 'dataset'
        approved = contract.read_json(contract.APPROVED_PATH)
        rows = []
        for index, expected in enumerate(approved):
            row = dict(expected, subject_id='mailbox')
            for field in ('image', 'mask'):
                path = source / 'assets' / row['id'] / (field + 's') / 'image.png'
                path.parent.mkdir(parents=True)
                image = Image.new('RGB', (512, 512), (index + 10, 50, 90)) if field == 'image' else Image.new('L', (512, 512), 0)
                if field == 'mask':
                    ImageDraw.Draw(image).rectangle((140, 320, 440, 511), fill=255)
                image.save(path)
                row[field], row[field + '_sha256'] = str(path), contract.sha256(path)
                expected[field + '_sha256'] = row[field + '_sha256']
            rows.append(row)
        approved_path = root / 'approved.json'
        approved_path.write_text(json.dumps(approved), encoding='utf8')
        source_manifest = source / 'training_assets_manifest.jsonl'
        # Source order is deliberately different; the reviewed order must win.
        source_manifest.write_text(''.join(json.dumps(row) + '\n' for row in reversed(rows)), encoding='utf8')
        with patch.object(contract, 'SOURCE_ROOT', source), patch.object(contract, 'APPROVED_PATH', approved_path), \
                patch.dict(os.environ, {contract.DATASET_ENV: str(output)}):
            yield source, output, rows


class Center11ContractTest(unittest.TestCase):
    def test_only_authorized_training_arguments_change(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {contract.DATASET_ENV: temporary}):
            for method in contract.METHODS:
                old = contract.baseline.build_config(method, Path(temporary))
                new = contract.build_config('subject', method)
                args = new['args']
                changed = {key for key in old['args'].keys() | args.keys() if old['args'].get(key) != args.get(key)}
                self.assertEqual(changed, {'pretrained_model_name_or_path', 'max_train_steps', 'checkpoint_steps'})
                self.assertEqual(args['pretrained_model_name_or_path'], str(contract.MODEL_DIR))
                self.assertEqual(args['max_train_steps'], 1500)
                self.assertEqual(args['checkpoint_steps'], [500, 1000, 1500])
                self.assertEqual((args['modifier_token'], args['initializer_token']), ('<S*>', 'mailbox'))
                self.assertEqual((args['resolution'], args['mixed_precision'], args['seed']), (512, 'no', 42))
                self.assertEqual(args['learning_rate'], 1e-5)
                self.assertEqual(args['kv_learning_rate'], 1e-5 if method == 'fullrank' else 5e-5)
                self.assertNotIn('resume_from_checkpoint', args)
                self.assertFalse(new['data_lineage']['old_weights_reused'])
                if method == 'fullrank':
                    self.assertTrue(args['token_local_kv'])
                else:
                    self.assertEqual((args['subject_lora_mode'], args['subject_lora_rank'], args['subject_lora_alpha']),
                                     ('token_local_kv', 64, 64))

    def test_approved_list_is_exact_center_then_three_photos(self):
        rows = contract.read_json(contract.APPROVED_PATH)
        expected = [f'center__{color}__M{m}_R{r}__studio'
                    for color in ('blue', 'green') for m in (0, 1) for r in (15, 85)]
        self.assertEqual([row['id'] for row in rows], expected + ['photo_original', 'photo_lch_blue', 'photo_lch_green'])
        self.assertEqual(len({row['image_sha256'] for row in rows}), 11)
        self.assertEqual(len({row['mask_sha256'] for row in rows[:8]}), 1)
        self.assertEqual(len({row['mask_sha256'] for row in rows[8:]}), 1)
        self.assertNotEqual(rows[0]['mask_sha256'], rows[8]['mask_sha256'])
        for row in rows:
            self.assertEqual(row['caption'].count('<S*>'), 1)
            for field in ('image_sha256', 'mask_sha256'):
                self.assertRegex(row[field], '^[0-9a-f]{64}$')

    def test_stage_preserves_bytes_order_and_one_caption_pairs(self):
        with staged_fixture() as (source, output, originals):
            before = {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()}
            result = contract.stage_dataset(output)
            self.assertEqual(result['rows'], 11)
            refs = result['references']
            self.assertEqual([r['id'] for r in refs], [r['id'] for r in originals])
            concepts = contract.read_json(output / 'concepts.json')
            self.assertEqual(len(concepts), 11)
            self.assertEqual(len({c['instance_data_dir'] for c in concepts}), 11)
            for row, original, concept in zip(refs, originals, concepts):
                self.assertEqual(concept['instance_prompt'], [row['caption']])
                self.assertEqual(concept['instance_data_dir'], str(Path(row['image']).parent))
                self.assertEqual(concept['instance_mask_dir'], str(Path(row['mask']).parent))
                for field in ('image', 'mask'):
                    self.assertEqual(Path(row[field]).read_bytes(), Path(original[field]).read_bytes())
            self.assertEqual(before, {str(p): p.read_bytes() for p in source.rglob('*') if p.is_file()})

    def test_stage_rejects_changed_source_bytes_or_caption(self):
        for mutation in ('image', 'caption'):
            with self.subTest(mutation=mutation), staged_fixture() as (source, output, rows):
                if mutation == 'image':
                    path = Path(rows[0]['image'])
                    path.write_bytes(path.read_bytes() + b'changed')
                else:
                    rows[0]['caption'] = 'a photo of <S*> mailbox in red color'
                    (source / 'training_assets_manifest.jsonl').write_text(
                        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
                with self.assertRaises(ValueError):
                    contract.stage_dataset(output)

    def test_verify_rejects_changed_source(self):
        with staged_fixture() as (_, output, rows):
            contract.stage_dataset(output)
            path = Path(rows[-1]['mask'])
            path.write_bytes(path.read_bytes() + b'changed')
            with self.assertRaises(ValueError):
                contract.verify_dataset('subject')

    def test_verify_checks_caption_pairing_even_with_updated_file_hash(self):
        with staged_fixture() as (_, output, _):
            contract.stage_dataset(output)
            path = output / 'concepts.json'
            concepts = contract.read_json(path)
            concepts[0]['instance_prompt'], concepts[1]['instance_prompt'] = concepts[1]['instance_prompt'], concepts[0]['instance_prompt']
            path.write_text(json.dumps(concepts), encoding='utf8')
            audit_path = output / 'dataset_audit.json'
            audit = contract.read_json(audit_path)
            audit['files'][str(path)] = contract.sha256(path)
            audit_path.write_text(json.dumps(audit), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'one-image one-caption'):
                contract.verify_dataset('subject')

    def test_verify_rejects_extra_training_image(self):
        with staged_fixture() as (_, output, _):
            result = contract.stage_dataset(output)
            path = Path(result['references'][0]['image'])
            path.with_name('extra.png').write_bytes(path.read_bytes())
            with self.assertRaisesRegex(ValueError, 'one-image/one-mask'):
                contract.verify_dataset('subject')


class Center11TransferTest(unittest.TestCase):
    def test_exact_594_single_subject_probes(self):
        all_rows = []
        for method in contract.METHODS:
            expected = [r['prompt'] for r in blue_green_prompt_rows(500) if r['method'] == method and r['seed'] == 42]
            self.assertEqual(len(expected), 9)
            for step in (500, 1000, 1500):
                rows = inference.prompt_rows(method, step)
                self.assertEqual(len(rows), 99)
                self.assertEqual(list(dict.fromkeys(row['prompt'] for row in rows)), expected)
                self.assertEqual({row['seed'] for row in rows}, set(range(30, 41)))
                self.assertEqual(len({row['image_path'] for row in rows}), 99)
                for row in rows:
                    self.assertEqual((row['branch'], row['method'], row['step'], row['cfg']), ('subject', method, step, 3.5))
                    self.assertTrue(row['subject_adapter_loaded'])
                    self.assertEqual(row['prompt'].count('<S*>'), 1)
                    self.assertFalse(any(token in row['prompt'] for token in ('<C*>', '<M*>', '<T*>')))
                all_rows.extend(rows)
        self.assertEqual(len(all_rows), 594)
        self.assertEqual(len({r['id'] for r in all_rows}), 594)
        for key, value in {'scheduler': 'PNDM', 'num_inference_steps': 100, 'guidance_scale': 3.5,
                           'width': 512, 'height': 512, 'dtype': 'float16', 'safety_checker': True}.items():
            self.assertEqual(inference.SAMPLING[key], value)

    def test_undeclared_step_and_method_rejected(self):
        for method, step in [('fullrank', 400), ('lora_r64', 600), ('fullrank', 500.0), ('other', 500)]:
            with self.subTest(method=method, step=step), self.assertRaises(ValueError):
                inference.prompt_rows(method, step)


class Center11IsolationTest(unittest.TestCase):
    def test_gpu0_guard_rejects_process_or_memory_without_fallback(self):
        for memory, processes in [(0, 'GPU-zero, 1234\n'), (513, '')]:
            with self.subTest(memory=memory, processes=processes):
                stats = f'0, GPU-zero, {memory}\n1, GPU-one, 0\n2, GPU-two, 0\n3, GPU-three, 0\n'
                with patch.object(run.subprocess, 'check_output', side_effect=[stats, processes]) as command:
                    with self.assertRaisesRegex(RuntimeError, 'GPU0 occupied'):
                        run.require_idle_gpu0()
                self.assertEqual(command.call_count, 2)
                self.assertTrue(all(c.args[0][0] == 'nvidia-smi' for c in command.call_args_list))

    def test_idle_gpu0_accepted_when_sdxl_gpus_busy(self):
        stats = '0, GPU-zero, 0\n1, GPU-one, 45000\n2, GPU-two, 45000\n3, GPU-three, 45000\n'
        with patch.object(run.subprocess, 'check_output', side_effect=[stats, 'GPU-one, 1\nGPU-two, 2\nGPU-three, 3\n']):
            run.require_idle_gpu0()

if __name__ == '__main__':
    unittest.main()
