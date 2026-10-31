"""Check wood180 mixed provenance and actual intermediate Subject embedding pairing."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.wood_spoon_composition_v1 import source, source180
from tests import test_wood_spoon_composition_source as fixtures


class Wood180SourceTest(unittest.TestCase):
    setUp = fixtures.WoodCompositionSourceTest.setUp
    fixture = fixtures.WoodCompositionSourceTest.fixture
    verify = fixtures.WoodCompositionSourceTest.verify

    def test_reject_unsupported_steps_before_any_source_access(self):
        for step in (500, 600, 750.0, True, None):
            with self.subTest(step=step), patch.object(source, 'prepare') as base:
                with self.assertRaisesRegex(ValueError, 'Subject750 or Subject1000'):
                    source180.prepare(self.root / 'unused', step)
                base.assert_not_called()

    def test_actual_subject750_uses_750_encoder_and_rejects_final1000_embedding(self):
        fixture = self.fixture('texture', 'lora_r64', 750)
        parent, derived, directory, payloads, _ = fixture
        checkpoint = Path(parent['checkpoint'])
        for folder in (checkpoint, directory):
            payloads[folder / '<S*>.bin'] = {'<S*>': payloads[folder / '<T*>.bin']['<T*>']}
            self.virtual_files[folder / '<S*>.bin'] = self.virtual_files[folder / '<T*>.bin']
            adaptation = json.loads((folder / 'adaptation_config.json').read_text())
            adaptation['modifier_tokens'] = ['<S*>']
            (folder / 'adaptation_config.json').write_text(json.dumps(adaptation))
        audit_path = checkpoint / 'embedding_update_audit.json'
        audit = json.loads(audit_path.read_text())
        audit['modifier_tokens'][0]['token'] = '<S*>'
        audit_path.write_text(json.dumps(audit))
        for row in (parent, derived):
            row.update(branch='subject', concept_id='wood_spoon', token='<S*>')
        parent['hashes'] = {path.replace('<T*>.bin', '<S*>.bin'):
            source.sha256(path.replace('<T*>.bin', '<S*>.bin')) for path in parent['hashes']}
        derived['source_hashes'] = copy.deepcopy(parent['hashes'])
        derived['hashes'] = {path.replace('<T*>.bin', '<S*>.bin'):
            source.sha256(path.replace('<T*>.bin', '<S*>.bin')) for path in derived['hashes']}
        derived['derived_token_sha256'] = source.sha256(directory / '<S*>.bin')
        # Historical Subject exports inherit the parent's flag even at750.
        derived['final_snapshot_matches_export'] = True
        (directory / 'receipt.json').write_text(json.dumps(derived))
        self.assertEqual(self.verify(fixture), derived)
        self.assertEqual(derived['training_steps'], 1000)
        payloads[directory / '<S*>.bin'] = payloads[checkpoint / '<S*>.bin']
        with self.assertRaisesRegex(ValueError, 'tensors differ'):
            self.verify(fixture)

    def test_selected750_loader_preserves_method_revisions_and_existing_export_paths(self):
        root = self.root / 'subject180'
        continuation = root / 'continuations/frozen/all180'
        output = self.root / 'audit'
        output.mkdir()
        parents = []
        for method in source.METHODS:
            run = (root / 'all180/training/fullrank/runs/original' if method == 'fullrank'
                   else continuation / 'training/lora_r64/runs/continued')
            parents.append({'method': method, 'branch': 'subject', 'concept_id': 'wood_spoon',
                'token': '<S*>', 'status': 'verified', 'steps': 1000, 'run': str(run)})
        index = {'git': source180.CONTINUATION_GIT, 'runs': parents,
            'source_gits': {'fullrank': source180.OLD_GIT, 'lora_r64': source180.CONTINUATION_GIT},
            'reuse_receipt': {'locks': {}}}
        files = {continuation / 'training_index.json': index,
            continuation / 'step750/provenance.json': {'git': source180.CONTINUATION_GIT,
                'training': parents, 'status': 'succeeded', 'step': 750,
                'training_images': 180, 'generated_images': 20}}
        for name in ('dataset_audit.json', 'source_locks.json', 'approval.json', 'concepts.json'):
            files[root / 'all180/training_dataset' / name] = {}
        for parent in parents:
            files[continuation / 'step750/checkpoints' / parent['method'] / 'receipt.json'] = {
                **parent, 'steps': 750, 'step': 750, 'training_steps': 1000}
        for path, data in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        process = SimpleNamespace(returncode=0, stdout='WOOD180_COMPOSITION_AUDIT=' +
            json.dumps({'parents': parents, 'git': source180.CONTINUATION_GIT}) + '\n')
        with patch.object(source180, 'SUBJECT_ROOT', root), \
             patch.object(source180, 'CONTINUATION_ROOT', continuation), \
             patch.object(source180.subprocess, 'run', return_value=process), \
             patch.object(source, '_references', return_value=[{'id': i} for i in range(180)]), \
             patch.object(source, 'verify_pair', side_effect=lambda parent, derived, *args: derived) as pairs:
            selected, references = source180._subject(output, 750, {})
            self.assertEqual(len(references), 180)
            self.assertEqual(set(selected), set(source.METHODS))
            for call in pairs.call_args_list:
                self.assertEqual(call.args[0]['steps'], 1000)
                self.assertEqual(call.args[1]['steps'], 750)
                self.assertEqual(call.args[2], 750)
                self.assertEqual(call.args[3], continuation / 'step750/checkpoints' / call.args[0]['method'])
            bad = copy.deepcopy(index)
            bad['source_gits']['fullrank'] = source180.CONTINUATION_GIT
            (continuation / 'training_index.json').write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'mixed-revision'):
                source180._subject(output, 750, {})

    def test_prepare_replaces_auxiliary72_with180_and_retains_texture500(self):
        inference_git = {'commit': 'cpu-inference', 'branch': 'cpu'}
        subject = {method: {'method': method, 'steps': 750, 'training_steps': 1000} for method in source.METHODS}
        texture = {method: {'method': method, 'steps': 500, 'training_steps': 1000} for method in source.METHODS}
        def base_prepare(output, **kwargs):
            self.assertEqual(kwargs, {'texture_step': 500, 'include_material': False})
            output.mkdir()
            result = {'sources': {'subject': {'old72': True}, 'texture': texture}, 'source_locks': {},
                'references': {'subject': list(range(72)), 'texture': list(range(16))},
                'training_gits': {'subject': source.SUBJECT_GIT, 'texture': source.NATURAL_GIT},
                'inference_git': inference_git}
            (output / 'audit.json').write_text(json.dumps(result))
            return result
        with patch.object(source180, 'git_info', return_value=inference_git), \
             patch.object(source180, 'resolve_run_dir', side_effect=Path), \
             patch.object(source, 'prepare', side_effect=base_prepare), \
             patch.object(source180, '_subject', return_value=(subject, list(range(180)))) as subject_audit:
            result = source180.prepare(self.root / 'combined', 750)
        self.assertEqual(result['sources'], {'subject': subject, 'texture': texture})
        self.assertEqual(len(result['references']['subject']), 180)
        self.assertEqual(result['training_gits']['subject'],
            {'fullrank': source180.OLD_GIT, 'lora_r64': source180.CONTINUATION_GIT})
        self.assertEqual(subject_audit.call_args.args[1], 750)
        self.assertTrue((self.root / 'combined/audit.json').is_file())
        auxiliary = json.loads((self.root / 'combined/base72_texture500/audit.json').read_text())
        self.assertEqual(auxiliary['sources']['subject'], {'old72': True})
        source.verify_locks(result['source_locks'])


if __name__ == '__main__':
    unittest.main()
