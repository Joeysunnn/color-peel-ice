import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[3] / "src" / "train" / "initializer_token_utils.py"
SPEC = importlib.util.spec_from_file_location("initializer_token_utils", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    def __init__(self, encodings):
        self.encodings = encodings

    def encode(self, text, add_special_tokens=True):
        if add_special_tokens:
            raise AssertionError("initializer validation must exclude special tokens")
        return self.encodings[text]


class InitializerTokenTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer(
            {
                "cyan": [1470, 550],
                "aqua": [18613],
                "teal": [22821],
                "turquoise": [19899],
            }
        )

    def test_rejects_multi_bpe_cyan_instead_of_truncating_it(self):
        with self.assertRaisesRegex(ValueError, "cyan.*exactly one token.*2"):
            MODULE.single_token_initializer_id(self.tokenizer, "cyan")

    def test_accepts_prevalidated_single_token_candidates(self):
        for candidate in ("aqua", "teal", "turquoise"):
            with self.subTest(candidate=candidate):
                token_id = MODULE.single_token_initializer_id(self.tokenizer, candidate)
                self.assertEqual(token_id, self.tokenizer.encodings[candidate][0])


if __name__ == "__main__":
    unittest.main()
