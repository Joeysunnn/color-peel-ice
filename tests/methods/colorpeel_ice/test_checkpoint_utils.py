import unittest

from src.train.checkpoint_utils import (
    should_save_checkpoint,
    tracker_safe_config,
    validate_checkpoint_plan,
)


class CheckpointUtilsTest(unittest.TestCase):
    def test_explicit_checkpoint_plan_saves_only_requested_steps(self):
        requested = [300, 500, 700, 1000, 2000]
        plan = validate_checkpoint_plan(requested, 1000, 2000)

        self.assertEqual(plan, frozenset(requested))
        self.assertEqual(
            [step for step in range(1, 2001)
             if should_save_checkpoint(step, plan, 1000)],
            requested,
        )

    def test_legacy_checkpoint_interval_is_unchanged(self):
        plan = validate_checkpoint_plan(None, 1000, 3000)

        self.assertIsNone(plan)
        self.assertEqual(
            [step for step in range(1, 3001)
             if should_save_checkpoint(step, plan, 1000)],
            [1000, 2000, 3000],
        )

    def test_checkpoint_plan_rejects_invalid_steps(self):
        invalid_plans = (
            [], [0, 300], [-1, 300], [True, 300],
            [300, 300], [500, 300], [300, 2001],
        )
        for steps in invalid_plans:
            with self.subTest(steps=steps):
                with self.assertRaisesRegex(ValueError, "checkpoint_steps"):
                    validate_checkpoint_plan(steps, 1000, 2000)

    def test_checkpoint_plan_rejects_nonpositive_legacy_interval(self):
        with self.assertRaisesRegex(ValueError, "checkpointing_steps"):
            validate_checkpoint_plan(None, 0, 2000)

    def test_tracker_config_serializes_checkpoint_steps_without_mutation(self):
        original = {"checkpoint_steps": [300, 500, 700], "max_train_steps": 700}

        self.assertEqual(tracker_safe_config(original), {
            "checkpoint_steps": "300,500,700",
            "max_train_steps": 700,
        })
        self.assertEqual(
            original,
            {"checkpoint_steps": [300, 500, 700], "max_train_steps": 700},
        )


if __name__ == "__main__":
    unittest.main()
