import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate_d1_subject_recolor_transfer.py"
PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_gorilla_transfer_generation_protocol_v1.json"
SPEC = importlib.util.spec_from_file_location("subject_transfer", SCRIPT)
subject_transfer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(subject_transfer)


def test_fixed_subject_transfer_manifest_has_all_user_prompts_and_85_requests():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = subject_transfer.build_manifest(protocol)
    assert len(rows) == 85
    assert {row["seed"] for row in rows} == {42, 43, 44, 45, 46}
    assert {row["group"] for row in rows} == {
        "identity_baseline", "seen_color_consistency", "unseen_color_transfer_primary",
        "background_context_transfer", "view_composition_transfer", "hard_diagnostic",
    }
    assert {row["prompt"] for row in rows} == {
        "a photo of <S*>", "a close-up photo of <S*>",
        "a photo of <S*> in red color", "a photo of <S*> in green color", "a photo of <S*> in blue color",
        "a photo of <S*> in purple color", "a photo of <S*> in orange color", "a photo of <S*> in pink color",
        "a photo of <S*> on a plain white background", "a photo of <S*> in a museum",
        "a photo of <S*> in a park", "a photo of <S*> on a pedestal",
        "a side view photo of <S*>", "a low-angle photo of <S*>", "a full-body photo of <S*>",
        "a photo of <S*> in purple color on a plain white background", "a side view photo of <S*> in orange color",
    }


def test_dry_run_writes_manifest_and_source_provenance(tmp_path):
    output = tmp_path / "transfer"
    assert subject_transfer.main(["--protocol", str(PROTOCOL), "--output-dir", str(output), "--dry-run"]) == 0
    assert len((output / "generation_manifest.jsonl").read_text(encoding="utf-8").splitlines()) == 85
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["source_training"]["subject_token"] == "<S*>"
    assert provenance["source_training"]["model_dir"].endswith("__d16c964__42/checkpoints")
    assert provenance["model_artifact_sha256"] is None
