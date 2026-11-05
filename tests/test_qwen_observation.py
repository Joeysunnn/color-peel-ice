from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from src.methods.colorpeel_ice.vlm_initialization import QwenPredictor, qwen_observation


class QwenObservationTests(unittest.TestCase):
    def test_large_nonsquare_images_keep_aspect_without_crop_or_source_mutation(self):
        for size, expected in (((2560, 1280), (1024, 512)), ((1280, 2560), (512, 1024))):
            with self.subTest(size=size):
                source = Image.linear_gradient('L').resize(size).convert('RGB')
                before = source.tobytes()
                observed = qwen_observation(source)
                self.assertEqual(observed.size, expected)
                self.assertEqual(observed.tobytes(), source.resize(expected, Image.Resampling.LANCZOS).tobytes())
                self.assertEqual(source.size, size)
                self.assertEqual(source.tobytes(), before)

    def test_small_and_limit_images_are_independent_pixel_identical_copies(self):
        for size in ((73, 41), (1024, 512)):
            with self.subTest(size=size):
                source = Image.linear_gradient('L').resize(size).convert('RGB')
                before = source.tobytes()
                observed = qwen_observation(source)
                self.assertIsNot(observed, source)
                self.assertEqual(observed.size, source.size)
                self.assertEqual(observed.tobytes(), before)
                observed.putpixel((0, 0), (255, 0, 0))
                self.assertEqual(source.tobytes(), before)

    def test_predictor_passes_observation_and_records_actual_processor_grid(self):
        class Inputs(dict):
            def to(self, _): return self
            @property
            def input_ids(self): return self['input_ids']

        grid = Mock()
        grid.detach.return_value.cpu.return_value.tolist.return_value = [[1, 32, 64]]
        inputs = Inputs(input_ids=np.zeros((1, 5)), image_grid_thw=grid)
        predictor = QwenPredictor.__new__(QwenPredictor)
        predictor.device, predictor.prompt, predictor.runtime = 'cpu', 'fixture prompt', {}
        predictor.torch = SimpleNamespace(inference_mode=nullcontext)
        predictor.processor = SimpleNamespace(apply_chat_template=Mock(return_value=inputs),
                                             batch_decode=Mock(return_value=['fixture response']))
        predictor.model = SimpleNamespace(generate=Mock(return_value=np.zeros((1, 7))))
        source = Image.new('RGB', (2560, 1280), 'orange')
        before = source.tobytes()
        self.assertEqual(predictor(source), 'fixture response')
        observed = predictor.processor.apply_chat_template.call_args.args[0][0]['content'][0]['image']
        self.assertEqual(observed.size, (1024, 512))
        self.assertEqual(source.tobytes(), before)
        self.assertEqual(predictor.runtime['image_observation']['original_size'], [2560, 1280])
        self.assertEqual(predictor.runtime['image_observation']['observed_size'], [1024, 512])
        self.assertTrue(predictor.runtime['image_observation']['resized'])
        self.assertEqual(predictor.runtime['image_grid_thw'], [[1, 32, 64]])
        inputs.pop('image_grid_thw')
        predictor(Image.new('RGB', (512, 512)))
        self.assertFalse(predictor.runtime['image_observation']['resized'])
        self.assertIsNone(predictor.runtime['image_grid_thw'])


if __name__ == '__main__':
    unittest.main()
