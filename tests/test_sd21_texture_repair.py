"""Bounded screening plan and numerical Texture-only residual scaling checks."""

import copy
from types import SimpleNamespace
import unittest

from experiments.sd21_texture_repair_v1 import run as screen

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from experiments.sd21_mailbox_sct_v1.attention import SCTTokenLocalKVAttnProcessor
    from experiments.sd21_texture_repair_v1.attention import scale_texture_residuals


class TextureScreenPlanTest(unittest.TestCase):
    def test_exactly96_unique_cases_with_matched_prompts_checkpoints_and_seeds(self):
        rows = screen.plan()
        self.assertEqual(len(rows), 96)
        self.assertEqual(len({row['id'] for row in rows}), 96)
        self.assertEqual(len({row['image_path'] for row in rows}), 96)
        self.assertEqual([(value['step'], value['key_scale'], value['value_scale'])
                          for value in screen.CONDITIONS.values()],
                         [(400, 1, 1), (600, 1, 1), (800, 1, 1), (1000, 1, 1),
                          (1000, .25, .25), (1000, .5, .5), (1000, 0, 1), (1000, 1, 0)])
        for method in ('fullrank', 'lora_r64'):
            for condition, setting in screen.CONDITIONS.items():
                for scene, suffix in screen.sct.SCENES.items():
                    selected = [row for row in rows if (row['method'], row['condition'], row['scene']) == (method, condition, scene)]
                    self.assertEqual([row['seed'] for row in selected], [31, 38])
                    self.assertEqual({row['prompt'] for row in selected}, {screen.sct.PROMPT + ' ' + suffix})
                    self.assertTrue(all(row['steps'] == {'subject': 1000, 'color': 150, 'texture': setting['step']} for row in selected))

    def test_selection_pairs_each_texture_adapter_and_embedding_without_changing_subject_color(self):
        item = lambda branch, step: {'branch': branch, 'step': step, 'checkpoint': f'{branch}/{step}',
                                     'token': branch, 'hashes': {f'{branch}/{step}/adapter': f'a{step}', f'{branch}/{step}/token': f'e{step}'}}
        audit = {'sources': {branch: {method: item(branch, step) for method in screen.sct.METHODS}
                             for branch, step in (('subject', 1000), ('color', 150))},
                 'texture_sources': {str(step): {method: item('texture', step) for method in screen.sct.METHODS}
                                     for step in (400, 600, 800, 1000)},
                 'training_gits': {'subject': 's', 'color': 'c', 'texture': 't'}, 'inference_git': 'new'}
        for method in screen.sct.METHODS:
            for condition, setting in screen.CONDITIONS.items():
                result = screen.selections(audit, method, condition)
                self.assertIs(result['sources']['subject'], audit['sources']['subject'][method])
                self.assertIs(result['sources']['color'], audit['sources']['color'][method])
                self.assertIs(result['sources']['texture'], audit['texture_sources'][str(setting['step'])][method])
                self.assertEqual(len(result['source_locks']), 6)
                self.assertEqual(result['training_gits'], audit['training_gits'])


@unittest.skipIf(torch is None, 'requires server torch')
class TextureScaleTest(unittest.TestCase):
    def test_fullrank_and_lora_scale_only_texture_kv_and_preserve_scale1_path(self):
        torch.manual_seed(42)
        attn = SimpleNamespace(to_k=torch.nn.Linear(64, 64, bias=False), to_v=torch.nn.Linear(64, 64, bias=False))
        hidden = torch.randn(2, 6, 64)
        masks = {branch: torch.zeros(2, 6, dtype=torch.bool) for branch in screen.sct.TOKENS}
        for position, mask in enumerate(masks.values()):
            mask[1, position] = True
        for method in screen.sct.METHODS:
            original = SCTTokenLocalKVAttnProcessor(64, 64, method)
            with torch.no_grad():
                for parameter in original.parameters():
                    parameter.normal_(std=.03)
            for key_scale, value_scale in ((1, 1), (.25, .25), (.5, .5), (0, 1), (1, 0)):
                with self.subTest(method=method, key=key_scale, value=value_scale):
                    processor = copy.deepcopy(original)
                    before = {projection: dict(getattr(processor, projection + '_residuals').items()) for projection in ('k', 'v')}
                    unet = SimpleNamespace(attn_processors={'cross': processor, 'self': SCTTokenLocalKVAttnProcessor(method=method)})
                    scale_texture_residuals(unet, key_scale, value_scale)
                    for projection, scale in (('k', key_scale), ('v', value_scale)):
                        layers = getattr(processor, projection + '_residuals')
                        for branch in ('subject', 'color'):
                            self.assertIs(layers[branch], before[projection][branch])
                        if scale == 1:
                            self.assertIs(layers['texture'], before[projection]['texture'])
                    actual = processor.project_kv(attn, hidden, masks)
                    for index, (projection, scale) in enumerate((('k', key_scale), ('v', value_scale))):
                        expected = getattr(attn, 'to_' + projection)(hidden)
                        for branch in screen.sct.TOKENS:
                            delta = getattr(original, projection + '_residuals')[branch](hidden)
                            if branch == 'texture':
                                delta = delta * scale
                            expected = torch.where(masks[branch].unsqueeze(-1), expected + delta, expected)
                        torch.testing.assert_close(actual[index], expected, rtol=0, atol=0)
                    if key_scale != 1 or value_scale != 1:
                        with self.assertRaisesRegex(ValueError, 'once'):
                            scale_texture_residuals(unet, .5, .5)


if __name__ == '__main__':
    unittest.main()
