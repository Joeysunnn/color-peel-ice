import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.mailbox_alpha_control_v1 import render


class StopBeforeGpu(Exception):
    pass


class MailboxAlphaControlRenderTests(unittest.TestCase):
    def test_one_or_multiple_assets_pass_preflight_with_four_views_each(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / 'inputs'
            inputs.mkdir()
            assets = [inputs / f'asset_{index}.pt' for index in range(2)]
            for asset in assets:
                asset.write_bytes(b'preflight-only native asset fixture')
            review = inputs / 'review.json'
            review.write_text(json.dumps({'camera': {'extrinsics': np.eye(4).tolist(),
                                                     'intrinsics': np.eye(3).tolist()}}))
            for count in (1, 2):
                with self.subTest(count=count), \
                        patch.object(render, 'resolve_run_dir', side_effect=lambda path: path.resolve()), \
                        patch.object(render, 'git_info', return_value={'commit': 'fixture'}), \
                        patch.object(render.subprocess, 'check_output', side_effect=[render.OFFICIAL, '']), \
                        patch('experiments.sd21_qwen_initializers_v1.run.require_idle') as idle, \
                        patch.dict(render.os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                        patch.object(render, 'write', side_effect=StopBeforeGpu) as write:
                    # The first status write precedes torch/CUDA imports and rendering.
                    with self.assertRaises(StopBeforeGpu):
                        render.run(assets[:count], [f'mesh_{index}' for index in range(count)],
                                   review, root / f'output_{count}')
                    idle.assert_called_once_with([0])
                    state = write.call_args.args[1]
                    self.assertEqual(state['expected_renders'], 4 * count)
                    self.assertEqual(state['completed_renders'], 0)
                    self.assertFalse(state['renderer']['learned_opacity_used'])

    def test_empty_assets_and_invalid_labels_remain_rejected_before_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / 'asset.pt'
            cases = [([], []), ([asset], []), ([asset, asset], ['same', 'same']),
                     ([asset], ['bad/label'])]
            for assets, labels in cases:
                with self.subTest(assets=assets, labels=labels), \
                        patch.object(render, 'resolve_run_dir', side_effect=lambda path: path.resolve()), \
                        patch.object(render, 'git_info') as git:
                    with self.assertRaisesRegex(ValueError, 'at least one asset'):
                        render.run(assets, labels, root / 'review.json', root / 'output')
                    git.assert_not_called()


if __name__ == '__main__':
    unittest.main()
