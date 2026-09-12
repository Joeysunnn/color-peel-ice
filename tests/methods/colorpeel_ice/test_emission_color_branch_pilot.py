from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.methods.colorpeel_ice import run_d1_emission_color_branch_pilot as runner
from src.methods.colorpeel_ice import emission_color_branch_pilot as pilot


class EmissionColorBranchPilotTests(unittest.TestCase):
    def _passing_rows(self):
        return [{**request, "e_ch": 0.0, "object_ratio_ok": True, "interior_ok": True, "eligible_ok": True,
                 "outside_mask_changed_pixel_count": 0, "rgb_clipping_pixel_count": 0} for request in pilot.pilot_requests()]

    def test_requests_are_exact_18_and_keep_target_constant_across_shape_view(self) -> None:
        requests = pilot.pilot_requests()
        self.assertEqual(len(requests), 18)
        self.assertEqual({row["shape"] for row in requests}, {"cube", "sphere", "cylinder"})
        self.assertEqual({row["view_index"] for row in requests}, {0, 8, 16})
        self.assertTrue(all(row["material"] == "Emission" and row["emission_strength"] == 1.0 for row in requests))
        self.assertTrue(all(0.0 <= row["target_h_degrees"] < 360.0 for row in requests))
        for stable_id in {row["stable_id"] for row in requests}:
            group = [row for row in requests if row["stable_id"] == stable_id]
            self.assertEqual(len(group), 9)
            self.assertEqual(len({tuple(row["socket_rgba"]) for row in group}), 1)
            self.assertTrue(all(0.0 <= channel <= 1.0 for row in group for channel in row["linear_rgb"]))

    def test_color_gate_and_safety_fail_closed(self) -> None:
        rows = self._passing_rows()
        self.assertTrue(pilot.summarize_measurements(rows)["overall_pass"])
        rows[0]["e_ch"] = 5.1
        self.assertFalse(pilot.summarize_measurements(rows)["overall_pass"])
        rows = self._passing_rows()
        rows[0]["outside_mask_changed_pixel_count"] = 1
        with self.assertRaises(pilot.EmissionColorBranchError):
            pilot.summarize_measurements(rows)

    def test_plan_binds_assets_requests_and_adapter_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root, assets = Path(temp) / "run", Path(temp) / "assets"
            for relative in runner.ASSETS.values():
                path = assets / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"asset")
            value = runner.plan(root, assets)
            self.assertEqual(len(value["requests"]), 18)
            self.assertEqual(runner._load_plan_contract(root)[0], value)
            plan_path = root / runner.PLAN_NAME
            tampered = json.loads(plan_path.read_text(encoding="utf-8"))
            tampered["requests"][0]["target_a"] = 0.0
            plan_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(pilot.EmissionColorBranchError):
                runner._load_plan_contract(root)


if __name__ == "__main__":
    unittest.main()
