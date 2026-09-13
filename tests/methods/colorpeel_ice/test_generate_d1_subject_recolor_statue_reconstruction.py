import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate_d1_subject_recolor_statue_reconstruction.py"
PROTOCOL = ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_subject_recolor_gorilla_statue_reconstruction_generation_protocol_v1.json"
SPEC = importlib.util.spec_from_file_location("statue_reconstruction", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_reconstruction_grid_is_45_training_prompt_images():
    rows = module.build_manifest(json.loads(PROTOCOL.read_text(encoding="utf-8")))
    assert len(rows) == 45
    assert {row["checkpoint_steps"] for row in rows} == {250, 500, 1000}
    assert {row["seed"] for row in rows} == {42, 43, 44, 45, 46}
    assert {row["prompt"] for row in rows} == {
        "a photo of <S*> statue in red color", "a photo of <S*> statue in green color", "a photo of <S*> statue in blue color",
    }
