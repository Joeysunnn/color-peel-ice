from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts" / "methods" / "colorpeel_ice" / "generate_d1_emission_color_transfer_all.py"
SPEC = importlib.util.spec_from_file_location("emission_transfer_all", SCRIPT)
all_transfer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(all_transfer)


def test_all_transfer_matrix_has_ten_templates_times_ten_seeds(tmp_path):
    rows = [{"category": "transfer", "transfer_template_index": index, "prompt": f"a <c1*> object {index}"}
            for index in range(10) for _ in range(60)]
    source = tmp_path / "manifest.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    protocol = copy.deepcopy(all_transfer.protocol(ROOT / "experiments" / "natural_image_subject_color_pilot" / "configs" / "d1_emission_color_transfer_orange_all100_protocol_v1.json"))
    protocol["source_manifest"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = all_transfer.build_manifest(source, protocol)
    assert len(manifest) == 100
    assert [row["transfer_template_index"] for row in manifest[::10]] == list(range(10))
    assert {row["seed"] for row in manifest} == set(range(42, 52))
    assert all("<C*>" in row["prompt"] for row in manifest)
