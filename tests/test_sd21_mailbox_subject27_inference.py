import unittest

from experiments.sd21_mailbox_subject27_v1 import inference
from experiments.subject_trellis_dataset_v1.inference import blue_green_prompt_rows


class MailboxSubject27InferenceTest(unittest.TestCase):
    def test_exact_historical_mailbox_probes_and_594_total(self):
        all_rows = []
        for method in ('fullrank', 'lora_r64'):
            expected = [r['prompt'] for r in blue_green_prompt_rows(500)
                        if r['method'] == method and r['seed'] == 42]
            self.assertEqual(len(expected), 9)
            for step in (200, 400, 600):
                rows = inference.prompt_rows(method, step)
                self.assertEqual(len(rows), 99)
                self.assertEqual(list(dict.fromkeys(row['prompt'] for row in rows)), expected)
                self.assertEqual({row['seed'] for row in rows}, set(range(30, 41)))
                self.assertTrue(all(row['step'] == step and row['method'] == method and row['cfg'] == 3.5
                           and row['subject_adapter_loaded'] and row['branch'] == 'subject'
                           and row['prompt'].count('<S*>') == 1 and 'mailbox' in row['prompt']
                           and not any(token in row['prompt'] for token in ('<C*>', '<M*>', '<T*>'))
                           for row in rows))
                self.assertEqual(len({row['image_path'] for row in rows}), 99)
                all_rows.extend(rows)
        self.assertEqual(len(all_rows), 594)
        self.assertEqual(len({row['id'] for row in all_rows}), 594)
        self.assertEqual(inference.SAMPLING['scheduler'], 'PNDM')
        self.assertEqual(inference.SAMPLING['num_inference_steps'], 100)
        self.assertEqual(inference.SAMPLING['guidance_scale'], 3.5)
        self.assertEqual(inference.SAMPLING['dtype'], 'float16')
        self.assertIs(inference.SAMPLING['safety_checker'], True)


    def test_invalid_method_and_steps_rejected(self):
        for method, step in [('invalid', 200), ('fullrank', 500), ('lora_r64', 1000),
                             ('fullrank', 2), ('fullrank', 200.0)]:
            with self.subTest(method=method, step=step):
                with self.assertRaises(ValueError):
                    inference.prompt_rows(method, step)
