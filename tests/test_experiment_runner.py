import json
import os
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY_ROOT / "scripts" / "launch" / "colorpeel_run.py"
SPEC = spec_from_file_location("colorpeel_run", RUNNER)
colorpeel_run = module_from_spec(SPEC)
SPEC.loader.exec_module(colorpeel_run)


class ExperimentRunnerTests(unittest.TestCase):
    def config_and_run_dir(self, root: Path, commit: str):
        config = root / "train.yaml"
        config.write_text(
            json.dumps(
                {
                    "stage": "train",
                    "run": {"study": "clevr_subject_color_3x3", "variant": "smoke", "seed": 42},
                    "args": {"concepts_list": "${COLORPEEL_DATA_ROOT}/concepts.json", "max_train_steps": 2},
                }
            ),
            encoding="utf-8",
        )
        run_id = f"20260822-120000__clevr_subject_color_3x3__smoke__{commit[:7]}__42"
        return config, root / "runs" / "clevr_subject_color_3x3" / run_id

    def test_dry_run_writes_provenance_and_manages_output(self):
        commit = "0123456789abcdef0123456789abcdef01234567"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, run_dir = self.config_and_run_dir(root, commit)
            environment = {
                **os.environ,
                "COLORPEEL_RUN_ROOT": str(root / "runs"),
                "COLORPEEL_DATA_ROOT": str(root / "data"),
            }
            with patch.dict(os.environ, environment, clear=True), patch.object(
                colorpeel_run,
                "git_info",
                return_value={"commit": commit, "branch": "test"},
            ):
                result = colorpeel_run.main(
                    ["--config", str(config), "--run-dir", str(run_dir), "--dry-run"]
                )
            self.assertEqual(result, 0)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "dry-run")
            self.assertIn(str(run_dir / "checkpoints"), manifest["command"])
            self.assertTrue((run_dir / "command.sh").is_file())
            self.assertTrue((run_dir / "environment.txt").is_file())

    def test_rejects_output_arguments_in_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.yaml"
            path.write_text(
                json.dumps(
                    {
                        "stage": "train",
                        "run": {"study": "study", "variant": "bad", "seed": 42},
                        "args": {"output_dir": "forbidden"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "launcher-managed"):
                colorpeel_run.read_config(path)

    def test_rejects_run_directory_outside_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(os.environ, {"COLORPEEL_RUN_ROOT": str(root / "runs")}, clear=True):
                with self.assertRaisesRegex(ValueError, "below COLORPEEL_RUN_ROOT"):
                    colorpeel_run.resolve_run_dir(root / "outside" / "run")


if __name__ == "__main__":
    unittest.main()
