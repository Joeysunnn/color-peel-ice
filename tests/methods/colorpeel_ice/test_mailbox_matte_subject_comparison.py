"""Keep the three S arms on identical prompts and seeds."""

from pathlib import Path

from scripts.methods.colorpeel_ice.generate_color_material_diagnostic import read_json
from scripts.methods.colorpeel_ice.generate_mailbox_matte_subject_comparison import manifest_rows


ROOT = Path(__file__).resolve().parents[3]


def test_mailbox_comparison_grid_is_matched():
    protocol = read_json(ROOT / "experiments/subject_material_composition_v1/protocols/mailbox_matte_subject_inference_v1.json")
    baseline = read_json(ROOT / protocol["baseline_protocol"])
    rows = manifest_rows(protocol, baseline)

    assert len(rows) == 108
    for arm in ("baseline", "matte_only", "balanced"):
        subset = [row for row in rows if row["arm"] == arm]
        assert len(subset) == 36
        assert {(row["group"], row["condition"], row["seed"], row["prompt"])
                for row in subset} == {
                    (row["group"], row["condition"], row["seed"], row["prompt"])
                    for row in rows if row["arm"] == "baseline"
                }
