"""Compare literal metal with selected M under the same literal-orange prompt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.methods.colorpeel_ice.generate_color_material_diagnostic import read_json, sha256
from scripts.methods.colorpeel_ice.generate_literal_orange_material_comparison import (
    comparison_rows, generate,
)


def control_rows(protocol_path: Path, run_root: Path):
    protocol = read_json(protocol_path)
    if (protocol.get("schema") != "literal_metal_control/v1"
            or protocol.get("objects") != ["cube", "sphere", "mug"]
            or protocol.get("seeds") != [42, 43, 44]
            or protocol.get("reference_prompt") != "a photo of a {object} with orange color and <M*> material"
            or protocol.get("control_prompt") != "a photo of a {object} with orange color and metal material"
            or protocol.get("sampling") != {"num_inference_steps": 100, "guidance_scale": 3.5}
            or protocol.get("safety_checker") != "enabled"):
        raise ValueError("literal metal control protocol differs")
    source_protocol = REPO_ROOT / protocol["source_protocol"]
    if sha256(source_protocol) != protocol["source_protocol_sha256"]:
        raise ValueError("source protocol differs")
    source, color_dir, material_dir, source_rows = comparison_rows(source_protocol, run_root)
    if (source["literal_prompt"] != protocol["reference_prompt"]
            or source["sampling"] != protocol["sampling"]
            or source["objects"] != protocol["objects"]
            or source["seeds"] != protocol["seeds"]):
        raise ValueError("reference settings differ")
    reference = run_root / protocol["reference_run_relative_to_COLORPEEL_RUN_ROOT"]
    if (sha256(reference / "generation_status.jsonl") != protocol["reference_status_sha256"]
            or read_json(reference / "provenance.json").get("protocol_sha256") != protocol["source_protocol_sha256"]
            or read_json(reference / "provenance.json").get("status") != "succeeded"):
        raise ValueError("reference run differs")
    statuses = [json.loads(line) for line in (reference / "generation_status.jsonl").read_text(encoding="utf-8").splitlines()]
    by_pair = {(row["object"], row["seed"]): row for row in statuses}
    if len(statuses) != 9 or len(by_pair) != 9:
        raise ValueError("expected nine reference rows")
    rows = []
    for source_row in source_rows:
        obj, seed = source_row["object"], source_row["seed"]
        previous = by_pair[obj, seed]
        path = reference / previous["image_path"]
        if previous["status"] != "ok" or sha256(path) != previous["image_sha256"]:
            raise ValueError(f"reference image differs: {obj}, {seed}")
        rows.append({"object": obj, "seed": seed,
                     "literal_prompt": protocol["control_prompt"].format(object=obj),
                     "literal_image_path": f"images/{obj}/literal_orange_literal_metal__seed{seed}.png",
                     "reference_prompt": protocol["reference_prompt"].format(object=obj),
                     "reference_image_path": str(path),
                     "reference_image_sha256": previous["image_sha256"]})
    return protocol, color_dir, material_dir, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol, color_dir, material_dir, rows = control_rows(args.protocol, Path(os.environ["COLORPEEL_RUN_ROOT"]))
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "comparison_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    provenance = {"protocol_sha256": sha256(args.protocol), "color_dir": str(color_dir),
                  "material_dir": str(material_dir), "status": "dry_run" if args.dry_run else "running"}
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.dry_run:
        generate(protocol, color_dir, material_dir, rows, args.output_dir, args.device)
        provenance["status"] = "succeeded"
        (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(rows)} literal-metal/reference pairs: {args.output_dir}")


if __name__ == "__main__":
    main()
