import json
import os
import sys
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

    def test_segmentation_stage_manages_masks_and_failure_status(self):
        run_dir = Path("/runs/example")
        outputs = colorpeel_run.managed_output_args("segment", run_dir)
        config = {
            "stage": "segment",
            "run": {"study": "study", "variant": "eval", "seed": 42},
            "args": {"manifest": "/data/generation.jsonl", "image-dir": "/data/images"},
        }

        command = colorpeel_run.build_command(config, run_dir, {})

        self.assertEqual(outputs["mask-dir"], str(run_dir / "evaluation" / "masks"))
        self.assertEqual(
            outputs["output"], str(run_dir / "evaluation" / "segmentation_status.jsonl")
        )
        self.assertIn("segment_grounded_sam.py", command[1])
        self.assertIn("--mask-dir", command)

    def test_qwen_stage_manages_prediction_jsonl(self):
        run_dir = Path("/runs/example")
        outputs = colorpeel_run.managed_output_args("predict_qwen", run_dir)
        config = {
            "stage": "predict_qwen",
            "run": {"study": "study", "variant": "eval", "seed": 42},
            "args": {"manifest": "/data/generation.jsonl", "image-dir": "/data/images"},
        }

        command = colorpeel_run.build_command(config, run_dir, {})

        self.assertEqual(
            outputs["output"], str(run_dir / "evaluation" / "qwen_predictions.jsonl")
        )
        self.assertIn("predict_qwen.py", command[1])
        self.assertNotIn("--mask-dir", command)

    def test_evaluation_configs_build_with_explicit_upstream_paths(self):
        configs_dir = (
            REPOSITORY_ROOT
            / "experiments"
            / "clevr_subject_color_3x3"
            / "configs"
        )
        environment = {
            "COLORPEEL_CHECKPOINT_DIR": "/previous/train/checkpoints",
            "COLORPEEL_GENERATION_DIR": "/previous/generate/inference",
            "COLORPEEL_MASK_DIR": "/previous/segment/evaluation/masks",
            "COLORPEEL_QWEN_PREDICTIONS": "/previous/qwen/evaluation/qwen_predictions.jsonl",
        }
        expected_stages = {
            "generate.yaml": "generate",
            "segment.yaml": "segment",
            "predict_qwen.yaml": "predict_qwen",
            "score_color.yaml": "score_color",
            "score_clevr.yaml": "score_clevr",
        }

        for filename, expected_stage in expected_stages.items():
            config = colorpeel_run.read_config(configs_dir / filename)
            command = colorpeel_run.build_command(
                config, Path("/runs/current"), environment
            )
            self.assertEqual(config["stage"], expected_stage)
            self.assertEqual(command[0], sys.executable)
            self.assertFalse(any("${COLORPEEL_" in token for token in command))

        generate = colorpeel_run.read_config(configs_dir / "generate.yaml")
        generate_command = colorpeel_run.build_command(
            generate, Path("/runs/current"), environment
        )
        self.assertIn("--skip-existing", generate_command)
        self.assertIn("CompVis/stable-diffusion-v1-4", generate_command)

        segment = colorpeel_run.read_config(configs_dir / "segment.yaml")
        self.assertEqual(segment["protocol"]["box_threshold"], 0.25)
        self.assertEqual(segment["protocol"]["text_threshold"], 0.25)
        self.assertTrue(segment["protocol"]["local_files_only"])

        qwen = colorpeel_run.read_config(configs_dir / "predict_qwen.yaml")
        self.assertEqual(qwen["protocol"]["torch_dtype"], "float16")
        self.assertFalse(qwen["protocol"]["do_sample"])
        self.assertEqual(qwen["protocol"]["max_new_tokens"], 128)

        cyan_train = colorpeel_run.read_config(configs_dir / "train_cyan_initializer.yaml")
        cyan_command = colorpeel_run.build_command(
            cyan_train,
            Path("/runs/current"),
            {
                **environment,
                "COLORPEEL_CONCEPTS_LIST": "/data/concepts.json",
            },
        )
        initializer_index = cyan_command.index("--initializer_token") + 1
        self.assertEqual(
            cyan_command[initializer_index],
            "cube+sphere+cylinder+red+turquoise+gray",
        )
        self.assertEqual(cyan_train["protocol"]["selected_candidate"], "turquoise")
        self.assertEqual(cyan_train["protocol"]["scientific_change"], "cyan_initializer_only")

    def test_mask_dir_is_managed_only_for_segmentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = {
                "run": {"study": "study", "variant": "eval", "seed": 42},
                "args": {"mask-dir": "/external/masks"},
            }
            segment_path = root / "segment.yaml"
            segment_path.write_text(
                json.dumps({**base, "stage": "segment"}), encoding="utf-8"
            )
            color_path = root / "score_color.yaml"
            color_path.write_text(
                json.dumps({**base, "stage": "score_color"}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "launcher-managed"):
                colorpeel_run.read_config(segment_path)
            self.assertEqual(
                colorpeel_run.read_config(color_path)["args"]["mask-dir"],
                "/external/masks",
            )


if __name__ == "__main__":
    unittest.main()
