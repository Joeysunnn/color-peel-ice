from __future__ import annotations

import importlib.util
import json
import copy
import hashlib
from pathlib import Path


ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate_d1_emission_color_transfer_quick.py"
SPEC = importlib.util.spec_from_file_location("emission_transfer_quick", SCRIPT)
quick = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(quick)


def _source_manifest(path: Path) -> tuple[Path, dict]:
    templates = (
        "a {color} bowl on the table", "a {color} bowling ball in a bowling alley",
        "a {color} plate on the table", "a {color} vase on the shelf",
        "a women wearing {color} pants", "a {color} teddy-bear in Time Square",
        "a {color} snooker ball on the table", "a {color} parrot perched on a tree",
        "a {color} sofa in living room", "a {color} rose blooming in a wooden pot",
    )
    rows = [{"category": "transfer", "transfer_template_index": index,
             "prompt": template.format(color=token)}
            for index, template in enumerate(templates)
            for token in ("<c1*>", "<c2*>", "<c3*>") for _ in range(20)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    protocol = copy.deepcopy(quick.protocol(ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_emission_color_transfer_orange_quick40_protocol_v1.json"))
    protocol["source_manifest"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, protocol


def test_user_manifest_selection_is_fixed_five_templates_times_eight_seeds(tmp_path):
    source, protocol = _source_manifest(tmp_path / "manifest.jsonl")
    rows = quick.build_manifest(source, protocol)
    assert len(rows) == 40
    assert [row["transfer_template_index"] for row in rows[::8]] == [0, 6, 3, 9, 8]
    assert {row["seed"] for row in rows} == set(range(42, 50))
    assert all("<C*>" in row["prompt"] and quick.OLD_COLOR_TOKEN.search(row["prompt"]) is None for row in rows)


def test_hash_mismatch_rejects_source_manifest(tmp_path):
    source, protocol = _source_manifest(tmp_path / "manifest.jsonl")
    source.write_text("{}\n", encoding="utf-8")
    try:
        quick.build_manifest(source, protocol)
    except ValueError as error:
        assert "hash" in str(error)
    else:
        raise AssertionError("tampered source manifest was accepted")
