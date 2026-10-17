import ast
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from experiments.texture_material_disentangled_v1 import train, run_phase1


class TrainingContractTests(unittest.TestCase):
    def test_eight_recipes_freeze_unrelated_parameters_and_pair_methods(self):
        recipes = train.recipes()
        self.assertEqual(len(recipes), 8)
        for recipe in recipes:
            pilot = train.build_config(recipe, Path("staging"), "pilot")
            smoke = train.build_config(recipe, Path("staging"), "smoke")
            args = pilot["args"]
            self.assertEqual(args["modifier_token"], "<T*>" if recipe["branch"] == "texture" else "<M*>")
            self.assertEqual(args["initializer_token"], "texture" if recipe["branch"] == "texture" else "metal")
            self.assertEqual(args["max_train_steps"], 300)
            self.assertEqual(args["adam_weight_decay"], 0)
            self.assertEqual(args["cos_weight"], 0)
            self.assertEqual(args["freeze_model"], "crossattn_kv")
            self.assertFalse(args["hflip"])
            self.assertNotIn("lambda_attention", args)
            self.assertNotIn("joint_two_object_binding", args)
            expected = {**args, "max_train_steps": 2, "checkpointing_steps": 2, "checkpoint_steps": [2]}
            self.assertEqual(smoke["args"], expected)
            if recipe["method"] == "lora_r64":
                self.assertNotIn("token_local_kv", args)
                self.assertEqual((args["subject_lora_rank"], args["subject_lora_alpha"]), (64, 64))
            else:
                self.assertTrue(args["token_local_kv"])

    def test_lora_allowlist_keeps_old_modes_and_only_admits_token_local_texture(self):
        tree = ast.parse((train.ROOT / "src/train/train_colorpeel.py").read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "valid_concept" for target in node.targets))
        expression = compile(ast.Expression(assignment.value), "allowlist", "eval")
        for token, initializer, mode, expected in [
                ("<S*>", "mailbox", "full_kv", True), ("<S*>", "mailbox", "token_local_kv", True),
                ("<M*>", "metal", "token_local_kv", True), ("<M*>", "metal", "full_kv", False),
                ("<T*>", "texture", "token_local_kv", True), ("<T*>", "texture", "full_kv", False),
                ("<T*>", "metal", "token_local_kv", False), ("<C*>", "texture", "token_local_kv", False)]:
            with self.subTest(token=token, mode=mode, initializer=initializer):
                self.assertEqual(eval(expression, {"lora_token": token, "lora_initializer": initializer,
                    "args": types.SimpleNamespace(subject_lora_mode=mode)}), expected)

    def test_failed_render_stops_before_staging_training_and_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "new"
            codes, commands = iter([0, 1]), []

            def popen(command, **kwargs):
                commands.append(command)
                code = next(codes)
                return types.SimpleNamespace(pid=123, wait=lambda: code)

            with (patch.object(run_phase1, "git_info", return_value={"commit": "a" * 40}),
                  patch.object(run_phase1, "resolve_run_dir", side_effect=lambda path: path),
                  patch.object(run_phase1.subprocess, "Popen", side_effect=popen)):
                with self.assertRaisesRegex(RuntimeError, "render"):
                    run_phase1.run(output, 3, [1, 2])
            self.assertEqual(len(commands), 2)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual([step["stage"] for step in status["steps"]], ["plan", "render"])
            self.assertEqual(status["phase2"], "not_started")
            self.assertEqual(status["phase3"], "not_started")


if __name__ == "__main__":
    unittest.main()
