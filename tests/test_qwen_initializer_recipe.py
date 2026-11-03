"""Catch branch recipe drift and bad caption pairing before a GPU run."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.sd21_qwen_initializers_v1 import contract, data


class RecipeTest(unittest.TestCase):
    def test_staged_mask_names_work_with_actual_training_pairing(self):
        class Tokenizer:
            def add_tokens(self, tokens):
                return len(tokens)

            def convert_tokens_to_ids(self, token):
                return list(contract.TOKENS.values()).index(token) + 10

            def encode(self, text, **kwargs):
                return [1] + [self.convert_tokens_to_ids(token) for token in contract.TOKENS.values() if token in text] + [2]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image, mask = root / 'original.png', root / 'mask.png'
            Image.new('RGB', (512, 512), 'orange').save(image)
            Image.new('L', (512, 512), 255).save(mask)
            inputs = {}
            prompts = {'subject': 'a photo of <S*> mailbox', 'color': 'a photo of cube in <C*> color',
                       'material': 'a photo of cube made of <M*> under soft light',
                       'texture': 'a photo of cube with <T*> texture'}
            for branch, count in contract.COUNTS.items():
                inputs[branch] = [{'id': str(index), 'caption': prompts[branch], 'image': str(image),
                                  'image_sha256': data.sha256(image), 'mask': None if branch == 'color' else str(mask),
                                  'mask_sha256': None if branch == 'color' else data.sha256(mask)} for index in range(count)]
            approved = root / 'approved.json'
            approved.write_text(json.dumps([{'id': row['id']} for row in inputs['subject']]), encoding='utf8')
            with patch.object(data, 'source_rows', return_value=(inputs, {str(image): data.sha256(image)})), \
                    patch.object(data, 'APPROVED_PATH', approved):
                result = data.prepare(root / 'dataset', {'source': {'image_sha256': contract.SOURCE_SHA},
                                                        'subject_category': 'mailbox'}, Tokenizer())
            self.assertEqual({branch: value['rows'] for branch, value in result.items()}, contract.COUNTS)

    def test_four_fresh_recipes_and_smoke_dose(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            words = {'subject': 'mailbox', 'color': 'orange', 'material': 'glossy', 'texture': 'plain'}
            (root / 'source_audit.json').write_text(json.dumps({'initializer_manifest': str(root / 'initializers.json'),
                'initializers': {'branches': {branch: {'initializer': word} for branch, word in words.items()}}}), encoding='utf8')
            for smoke in (False, True):
                for branch in contract.BRANCHES:
                    config = contract.build_config(root, branch, smoke=smoke)
                    args = config['args']
                    self.assertEqual(args['initializer_token'], words[branch])
                    self.assertEqual(args['modifier_token'], contract.TOKENS[branch])
                    self.assertEqual(args['checkpoint_steps'], [2] if smoke else list(contract.STEPS[branch]))
                    self.assertEqual(args['max_train_steps'], args['checkpoint_steps'][-1])
                    self.assertEqual(args['pretrained_model_name_or_path'], str(contract.MODEL_DIR))
                    self.assertEqual(args['learning_rate'], 1e-5)
                    self.assertEqual(args['kv_learning_rate'], 5e-5)
                    self.assertEqual(args['subject_lora_mode'], 'token_local_kv')
                    self.assertEqual(args['subject_lora_rank'], 64)
                    self.assertEqual(args['adam_weight_decay'], 0)
                    self.assertEqual(args['concept_isolation_weight'], 0)
                    self.assertFalse(args.get('resume_from_checkpoint'))
                    self.assertEqual(args['concepts_list'], str(root / 'dataset' / branch / 'concepts.json'))

    def test_finish_caption_changes_only_declared_phrase(self):
        row = {'prompt': 'a photo of a red sphere made of <M*> under soft light'}
        self.assertEqual(data.caption(row, 'material', 'mailbox'),
                         'a photo of a red sphere with a <M*> surface under soft light')
        for branch, text in [('color', 'a photo of sphere shape in <C*> color'),
                             ('texture', 'a photo of a red cube with <T*> texture made of metal')]:
            self.assertEqual(data.caption({'prompt': text}, branch, 'mailbox'), text)
        with self.assertRaises(ValueError):
            data.caption({'prompt': 'a photo of <M*>'}, 'material', 'mailbox')

    def test_concepts_keep_caption_list_and_image_mask_separate(self):
        rows = [{'caption': 'a photo of <S*> mailbox', 'image': '/tmp/images/image.png', 'mask': '/tmp/masks/mask.png'}]
        result = data.concepts(rows)[0]
        self.assertEqual(result['instance_prompt'], ['a photo of <S*> mailbox'])
        self.assertNotEqual(result['instance_data_dir'], result['instance_mask_dir'])


if __name__ == '__main__':
    unittest.main()
