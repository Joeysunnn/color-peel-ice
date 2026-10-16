import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.sd15_token_local_kv_v1.run_full import make_queues, run_queue
from scripts.launch.colorpeel_run import resolve_run_dir


class FullTrainingTests(unittest.TestCase):
    def test_all_full_recipes_are_queued_once_and_preserve_dose(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary) / "20261016-150000__full_batch"
            queues = make_queues(batch, [1, 2, 3], "a" * 40)
            jobs = [job for queue in queues.values() for job in queue]
            self.assertEqual(len(jobs), 6)
            self.assertEqual(len({job["recipe"] for job in jobs}), 6)
            self.assertEqual([len(queue) for queue in queues.values()], [2, 2, 2])
            self.assertEqual({job["recipe"]: job["max_steps"] for job in jobs}, {
                "subject_fullrank": 2000, "subject_lora_r64": 3000,
                "material_ground_reflection_fullrank": 5000,
                "material_mailbox_lora_r64": 3000, "material_metal_spoon_lora_r64": 3000,
                "material_wood_spoon_lora_r64": 3000})
            self.assertTrue(all("_smoke2" not in job["config"] for job in jobs))

    def test_output_override_keeps_historical_data_root_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = {"COLORPEEL_RUN_ROOT": str(root / "old"),
                           "COLORPEEL_OUTPUT_ROOT": str(root / "new")}
            with patch.dict(os.environ, environment):
                self.assertEqual(resolve_run_dir(root / "new/run"), (root / "new/run").resolve())
                self.assertEqual(os.environ["COLORPEEL_RUN_ROOT"], str(root / "old"))
                with self.assertRaises(ValueError):
                    resolve_run_dir(root / "old/run")

    def test_failed_job_does_not_start_next_job_or_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            jobs = [{"recipe": name, "command": ["unused"], "status": "queued"}
                    for name in ("first", "second")]
            with patch("experiments.sd15_token_local_kv_v1.run_full.shutil.disk_usage") as disk:
                disk.return_value.free = 100 * (1 << 30)
                with patch("experiments.sd15_token_local_kv_v1.run_full.subprocess.Popen") as start:
                    start.return_value.pid = 123
                    start.return_value.wait.return_value = 1
                    self.assertFalse(run_queue(1, jobs, Path(temporary), {}))
                    self.assertEqual(start.call_count, 1)
            self.assertEqual([job["status"] for job in jobs], ["failed", "not_run"])

    def test_low_disk_space_stops_before_starting_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            jobs = [{"recipe": "first", "command": ["unused"], "status": "queued"}]
            with patch("experiments.sd15_token_local_kv_v1.run_full.shutil.disk_usage") as disk:
                disk.return_value.free = 20 * (1 << 30)
                with patch("experiments.sd15_token_local_kv_v1.run_full.subprocess.Popen") as start:
                    self.assertFalse(run_queue(1, jobs, Path(temporary), {}))
                    start.assert_not_called()
            self.assertEqual(jobs[0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
