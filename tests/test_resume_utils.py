import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "resume_utils", Path(__file__).resolve().parents[1] / "src/train/resume_utils.py")
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "parent/checkpoint-600"
        self.checkpoint.mkdir(parents=True)
        self.output = self.root / "continuation"
        self.output.mkdir()
        for name in resume.STATE_FILES:
            (self.checkpoint / name).write_bytes(b"fixture state")
        self.metrics = self.checkpoint.parent / "training_metrics.jsonl"
        rows = [dict(step=step, present_modifier_tokens=["<S*>"], losses_finite={"all": True},
                     **{key: 0.1 for key in resume.METRIC_FIELDS}) for step in range(1, 601)]
        self.metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.audit_path = self.checkpoint.parent / "embedding_update_audit.json"
        self.audit = {
            "observed_optimization_steps": 600,
            "modifier_tokens": [{"token": "<S*>", "token_id": 100,
                                 "exposure_steps": 600, "nonzero_gradient_steps": 599,
                                 "initial_final_l2_delta": 0.75}],
            "non_modifier_embedding_drift": {"changed_rows": 0},
        }
        self.audit_path.write_text(json.dumps(self.audit), encoding="utf-8")

    def plan(self, **overrides):
        kwargs = dict(num_processes=1, batch_size=1, accumulation=1, scheduler="constant", warmup=0)
        kwargs.update(overrides)
        return resume.prepare_resume(self.checkpoint, self.output, {"<S*>": 100}, 27, 1000, **kwargs)

    def test_final600_resumes_at601_and_seventh_sample(self):
        plan = self.plan()
        self.assertEqual((plan["global_step"], plan["first_epoch"], plan["skip_batches"]), (600, 22, 6))
        samples = [batch for epoch in range(plan["first_epoch"], 38) for batch in range(27)
                   if not (epoch == plan["first_epoch"] and batch < plan["skip_batches"])][:400]
        self.assertEqual(samples, [step % 27 for step in range(600, 1000)])
        self.assertEqual(len(plan["source_sha256"]), 8)

    def test_restore_real_counters_preserves_baseline_and_parent(self):
        plan = self.plan()
        baseline = object()
        original = self.metrics.read_bytes()

        class Audit:
            initial_embeddings = baseline
            exposure_steps = {"<S*>": 0}
            nonzero_gradient_steps = {"<S*>": 0}
            observed_optimization_steps = 0

            def finalize(inner, embeddings):
                return copy.deepcopy(self.audit)

        audit = Audit()
        resume.restore_resume_audit(plan, audit, None, self.output)
        self.assertIs(audit.initial_embeddings, baseline)
        self.assertEqual(audit.observed_optimization_steps, 600)
        self.assertEqual(audit.exposure_steps["<S*>"], 600)
        self.assertEqual(audit.nonzero_gradient_steps["<S*>"], 599)
        self.assertEqual(self.metrics.read_bytes(), original)
        self.assertEqual((self.output / "training_metrics.jsonl").read_bytes(), original)
        receipt = json.loads((self.output / "resume_receipt.json").read_text())
        self.assertEqual(receipt["first_new_step"], 601)
        self.assertIn("worker RNG", receipt["rng_scope"])

    def test_wrong_restored_embedding_rejected_before_output(self):
        plan = self.plan()

        class Audit:
            def finalize(inner, embeddings):
                result = copy.deepcopy(self.audit)
                result["modifier_tokens"][0]["initial_final_l2_delta"] = 0.1
                return result

        with self.assertRaisesRegex(ValueError, "restored embeddings"):
            resume.restore_resume_audit(plan, Audit(), None, self.output)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_missing_optimizer_rejected(self):
        (self.checkpoint / "optimizer.bin").unlink()
        with self.assertRaisesRegex(ValueError, "missing resume"):
            self.plan()

    def test_nonfinal_checkpoint_and_gapped_metrics_rejected(self):
        self.metrics.write_text("\n".join(self.metrics.read_text().splitlines()[:-1]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "completed final step"):
            self.plan()

    def test_bad_audit_count_rejected(self):
        self.audit["modifier_tokens"][0]["exposure_steps"] = 599
        self.audit_path.write_text(json.dumps(self.audit), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.plan()

    def test_nan_metrics_rejected(self):
        self.metrics.write_text(self.metrics.read_text().replace('"total_loss": 0.1', '"total_loss": NaN', 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid losses"):
            self.plan()

    def test_unsupported_protocol_rejected(self):
        for options in ({"num_processes": 2}, {"batch_size": 2}, {"accumulation": 2},
                        {"scheduler": "linear"}, {"warmup": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.plan(**options)

    def test_parent_output_and_existing_output_rejected(self):
        self.output = self.checkpoint.parent
        with self.assertRaisesRegex(ValueError, "immutable parent"):
            self.plan()
        self.output = self.root / "continuation"
        (self.output / "training_metrics.jsonl").write_text("existing")
        with self.assertRaisesRegex(ValueError, "already contains"):
            self.plan()

    def test_loaded_optimizer_scheduler_counters(self):
        plan = self.plan()

        class Optimizer:
            state = {0: {"step": 600, "exp_avg": object(), "exp_avg_sq": object()}}

        class Scheduler:
            epoch = 600

            def state_dict(self):
                return {"last_epoch": self.epoch, "_step_count": self.epoch + 1}

            def get_last_lr(self):
                return [0.1]

        optimizer, scheduler = Optimizer(), Scheduler()
        resume.validate_loaded_state(plan, optimizer, scheduler)
        self.assertEqual(plan["loaded_state"]["optimizer_step"], 600)
        scheduler.epoch = 0
        with self.assertRaisesRegex(ValueError, "scheduler counters"):
            resume.validate_loaded_state(plan, optimizer, scheduler)
        scheduler.epoch = 600
        optimizer.state = {}
        with self.assertRaisesRegex(ValueError, "state is empty"):
            resume.validate_loaded_state(plan, optimizer, scheduler)
        optimizer.state = {0: {"step": 0, "exp_avg": object(), "exp_avg_sq": object()}}
        with self.assertRaisesRegex(ValueError, "AdamW state"):
            resume.validate_loaded_state(plan, optimizer, scheduler)


class AcceleratorResumeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("accelerate"),
                         "requires the server Torch/Accelerate environment")
    def test_cpu_full_state_restores_weights_adam_scheduler_and_rng(self):
        import torch
        from accelerate import Accelerator

        accelerator = Accelerator(cpu=True)
        model = torch.nn.Linear(4, 4)
        embedding = torch.nn.Embedding(4, 4)
        accelerator.register_for_checkpointing(model)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(embedding.parameters()), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        model, embedding, optimizer, scheduler = accelerator.prepare(model, embedding, optimizer, scheduler)
        for _ in range(2):
            loss = model(embedding(torch.tensor([0, 1]))).square().mean()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
        table = embedding.weight.detach().clone()
        with tempfile.TemporaryDirectory() as temporary:
            accelerator.save_state(temporary)
            expected_rng = torch.randn(5)
            with torch.no_grad():
                embedding.weight.zero_()
                for value in model.parameters():
                    value.zero_()
            for state in optimizer.state.values():
                state["step"].zero_()
            scheduler.step()
            accelerator.load_state(temporary)
            self.assertTrue(torch.equal(torch.randn(5), expected_rng))
            self.assertTrue(torch.equal(embedding.weight, table))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, weights[name]))
            plan = {"global_step": 2, "metrics_text": json.dumps({"learning_rate": 0.01})}
            resume.validate_loaded_state(plan, optimizer, scheduler)
            self.assertEqual(plan["loaded_state"]["scheduler_last_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
