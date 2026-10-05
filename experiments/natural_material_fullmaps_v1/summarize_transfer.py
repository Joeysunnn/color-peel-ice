"""Summarize paired full-map transfer images and blind appearance labels."""

import argparse
import collections
import hashlib
import json
from pathlib import Path


MATERIALS = ("mailbox", "metal_spoon", "wood_spoon")
ARMS = ("base", "literal", "token", "subject_only", "subject_material")
SOURCE_COLORS = {"mailbox": {"red", "orange"},
                 "metal_spoon": {"gray", "grey", "silver"},
                 "wood_spoon": {"tan_brown", "tan", "brown", "light_brown"}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {"method": "blind visible-appearance labels, not physical A/R/M measurement",
              "materials": {}}
    for material in MATERIALS:
        evaluation = args.run / "transfer" / material
        audit = args.run / "appearance_audit" / material
        paths = {"manifest": evaluation / "manifest.jsonl",
                 "status": evaluation / "status.jsonl",
                 "evaluation_provenance": evaluation / "provenance.json",
                 "predictions": audit / "predictions.jsonl",
                 "audit_provenance": audit / "provenance.json"}
        provenance = json.loads(paths["evaluation_provenance"].read_text())
        audit_provenance = json.loads(paths["audit_provenance"].read_text())
        if provenance["status"] != "succeeded" or audit_provenance["status"] != "complete":
            raise ValueError(f"Incomplete evaluation: {material}")
        manifest, status, predictions = (read_jsonl(paths[name]) for name in
                                         ("manifest", "status", "predictions"))
        if len(manifest) != len(status) or len(manifest) != len(predictions) or len(manifest) != 24:
            raise ValueError(f"Unexpected row count: {material}")
        by_id = {row["id"]: row for row in manifest}
        if (len(by_id) != 24 or {row["id"] for row in status} != set(by_id)
                or {row["id"] for row in predictions} != set(by_id)):
            raise ValueError(f"Mismatched evaluation rows: {material}")
        for row in status + predictions:
            if row["image_sha256"] != sha(evaluation / "images" / f'{row["id"]}.png'):
                raise ValueError(f"Image hash mismatch: {material}/{row['id']}")
        filtered = {row["id"] for row in status if row["safety_filtered"]}
        arms = {}
        for arm in ARMS:
            selected = [row for row in predictions if row["arm"] == arm]
            valid = [row for row in selected if row["prediction"] is not None
                     and row["id"] not in filtered]
            arms[arm] = {
                "rows": len(selected), "valid": len(valid),
                "safety_filtered": sum(row["id"] in filtered for row in selected),
                "parse_errors": sum(row["prediction"] is None for row in selected),
                "schema_deviations": sum(bool(row["schema_deviations"]) for row in valid),
                "shape_match": sum(row["prediction"]["object_kind"] ==
                                   ("mailbox" if row["object"] == "subject_mailbox"
                                    else row["object"]) for row in valid),
                "source_color_match": sum(row["prediction"]["dominant_color"] in
                                          SOURCE_COLORS[material] for row in valid),
                "wood_grain_visible": sum(row["prediction"]["wood_grain_visible"] == "yes"
                                          for row in valid),
                "metallic_appearance": dict(sorted(collections.Counter(
                    row["prediction"]["metallic_appearance"] for row in valid).items())),
                "gloss": dict(sorted(collections.Counter(
                    row["prediction"]["gloss"] for row in valid).items())),
            }
        if [arms[arm]["rows"] for arm in ARMS] != [6, 6, 6, 3, 3]:
            raise ValueError(f"Unexpected arms: {material}")
        result["materials"][material] = {
            "evaluation_provenance": provenance,
            "audit_provenance_sha256": sha(paths["audit_provenance"]),
            "input_sha256": {name: sha(path) for name, path in paths.items()},
            "arms": arms,
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
