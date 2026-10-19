"""Keep the Material initializer ablation inside the existing token-local mode."""

import ast
from pathlib import Path
import types
import unittest


class MaterialInitializerGateTests(unittest.TestCase):
    def test_allowlist_adds_material_and_wood_for_token_local_material(self):
        root = Path(__file__).resolve().parents[3]
        tree = ast.parse((root / "src/train/train_colorpeel.py").read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "valid_concept"
                                  for target in node.targets))
        expression = compile(ast.Expression(assignment.value), "allowlist", "eval")
        for token in ("<S*>", "<M*>", "<T*>", "<C*>", None):
            for initializer in ("mailbox", "metal", "material", "texture", "wood", None):
                for mode in ("token_local_kv", "full_kv", None):
                    expected = ((token, initializer) == ("<S*>", "mailbox")
                                or (mode == "token_local_kv" and (
                                    token == "<M*>" and initializer in ("metal", "material", "wood")
                                    or (token, initializer) == ("<T*>", "texture"))))
                    with self.subTest(token=token, initializer=initializer, mode=mode):
                        actual = eval(expression, {"lora_token": token,
                            "lora_initializer": initializer,
                            "args": types.SimpleNamespace(subject_lora_mode=mode)})
                        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
