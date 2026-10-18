"""Compare completed old/new Material probes without modifying either source."""

import argparse
import csv
import html
import itertools
import json
import os
from pathlib import Path
import random

from PIL import Image, ImageDraw

from experiments.texture_material_disentangled_v1 import material_probe as probe


def verify_probe(root, protocol):
    paths = [root / name for name in ("summary.json", "provenance.json",
             "generation_manifest.jsonl", "generation_status.jsonl")]
    locks = {str(path): probe.sha256(path) for path in paths}
    summary, provenance = map(probe.read_json, paths[:2])
    step = provenance.get("training_steps")
    sampling = {**protocol["inference"], "conditions": list(probe.CONDITIONS)}
    if (summary.get("status") != "succeeded" or provenance.get("status") != "succeeded"
            or step not in probe.STEPS or summary.get("training_steps") != step
            or summary.get("expected_images") != 96 or summary.get("generated_images") != 96
            or summary.get("missing_images") != 0 or provenance.get("sampling") != sampling
            or provenance.get("texture_adapter_loaded") is not False):
        raise ValueError("probe is incomplete or its step/sampling differs")
    expected = {row["id"]: row for row in probe.build_rows(protocol, step)}
    tables = []
    for path in paths[2:]:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != 96 or {row["id"] for row in rows} != set(expected):
            raise ValueError("probe requires all 96 unique matched rows")
        for row in rows:
            if any(row.get(key) != value for key, value in expected[row["id"]].items()):
                raise ValueError("probe prompt, seed or condition differs")
        tables.append({row["id"]: row for row in rows})
    for row in tables[1].values():
        path = root / row["image_path"]
        digest = probe.sha256(path)
        if (row.get("status") not in ("ok", "safety_filtered")
                or row.get("image_sha256") != digest
                or row.get("material_adapter_loaded") != (row["condition"] == "learned_material")):
            raise ValueError("probe image or generation status differs")
        with Image.open(path) as image:
            if image.size != (sampling["resolution"], sampling["resolution"]):
                raise ValueError("probe image resolution differs")
            image.verify()
        locks[str(path)] = digest
    return step, tables[1], locks


def validate_probe(path):
    root = Path(path).resolve()
    protocol = probe.data.read_protocol()
    _, rows, locks = verify_probe(root, protocol)
    return protocol, rows, probe.read_json(root / "provenance.json"), locks


def run(old_probe_step, new_probe_step, output):
    old, new, output = (Path(path).resolve() for path in (old_probe_step, new_probe_step, output))
    if (old == new or output.exists() or any(output.is_relative_to(root) for root in (old, new, probe.ROOT))
            or old.is_relative_to(output) or new.is_relative_to(output)):
        raise ValueError("comparison output must be new and outside both sources and repository")
    protocol = probe.data.read_protocol()
    old_step, old_rows, old_locks = verify_probe(old, protocol)
    step, new_rows, new_locks = verify_probe(new, protocol)
    if old_step != step:
        raise ValueError("old/new training steps differ")
    locks = {**old_locks, **new_locks}
    output.mkdir(parents=True)
    (output / "contact_sheets").mkdir()
    sheets = []
    for method, material, seed in itertools.product(probe.infer.METHODS,
            probe.infer.target_ids(protocol, "material"), protocol["inference"]["seeds"]):
        sheet = Image.new("RGB", (6 * 272, 56 + 4 * 294), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"{method} / {material} / seed{seed} / step{step}", fill="black")
        for column, (label, root, rows, condition) in enumerate(
                (label, root, rows, condition) for label, root, rows in
                (("old", old, old_rows), ("uniform", new, new_rows)) for condition in probe.CONDITIONS):
            draw.text((8 + 272 * column, 32), f"{label}: {condition}", fill="black")
            for line, context in enumerate(protocol["dataset"]["heldout_contexts"]):
                identity = f"{method}__{material}__{context['id']}__{condition}__seed{seed}"
                row = rows[identity]
                x, y = 8 + 272 * column, 56 + 294 * line
                with Image.open(root / row["image_path"]) as image:
                    sheet.paste(image.convert("RGB").resize((256, 256)), (x, y))
                draw.text((x, y + 258), context["id"], fill="black")
                if row["status"] == "safety_filtered":
                    draw.text((x, y + 272), "SAFETY FILTERED", fill="red")
        path = f"contact_sheets/{method}__{material}__seed{seed}.jpg"
        sheet.save(output / path, quality=90)
        sheets.append(path)
    learned = [row for row in old_rows.values() if row["condition"] == "learned_material"]
    rng = random.Random(protocol["seed"])
    rng.shuffle(learned)
    key, cards = [], []
    with (output / "paired_blind_ratings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "material_appearance_preference",
            "fewer_extra_patterns", "shape_color_lighting_preservation", "notes"])
        writer.writeheader()
        for number, row in enumerate(learned):
            pair_id = f"pair_{number:03d}"
            sources = [("old", old, row), ("uniform", new, new_rows[row["id"]])]
            rng.shuffle(sources)
            excluded = any(item[2]["status"] == "safety_filtered" for item in sources)
            key.append({"pair_id": pair_id, "id": row["id"], "excluded_safety_filtered": excluded,
                        "A": sources[0][0], "B": sources[1][0]})
            if excluded:
                continue
            writer.writerow({"pair_id": pair_id})
            images = "".join(f'<figure><img width="256" src="{html.escape(os.path.relpath(root / item["image_path"], output).replace(chr(92), "/"), quote=True)}"><figcaption>{label}</figcaption></figure>'
                             for label, (_, root, item) in zip(("A", "B"), sources))
            cards.append(f'<section><h3>{pair_id}</h3>{images}</section>')
    probe.write_json(output / "paired_blind_key.json", {"pairs": key})
    (output / "paired_blind_review.html").write_text('<!doctype html><meta charset="utf-8">'
        '<title>Paired Material review</title><p>Rate A/B/tie in paired_blind_ratings.csv before opening the key. '
        'Assess material appearance, unwanted patterns and preservation of shape/color/lighting. '
        'Filtered pairs are excluded.</p>' + "".join(cards), encoding="utf-8")
    controls = [row for row in old_rows.values() if row["condition"] != "learned_material"]
    report = {"status": "succeeded", "training_steps": step, "old_source": str(old), "new_source": str(new),
        "source_locks": locks, "sampling": {**protocol["inference"], "conditions": list(probe.CONDITIONS)},
        "matched_pairs": 96, "learned_material_pairs": 32,
        "blind_review_pairs": len(cards), "safety_filtered_pairs": sum(
            row["status"] == "safety_filtered" or new_rows[row["id"]]["status"] == "safety_filtered"
            for row in old_rows.values()), "control_pairs": len(controls),
        "control_image_hashes_equal": sum(row["image_sha256"] == new_rows[row["id"]]["image_sha256"]
                                          for row in controls), "contact_sheets": sheets,
        "human_ratings": "pending", "texture_leakage_score": None, "physical_brdf_score": None,
        "limitations": ["Matched controls need not be bitwise identical across GPU runs.",
            "Visible highlights and reflections are not automatically albedo patterns.",
            "No visual quality or successful disentanglement is inferred from completion counts."]}
    if any(probe.sha256(Path(path)) != digest for path, digest in locks.items()):
        raise ValueError("source artifacts changed during comparison")
    probe.write_json(output / "report.json", report)
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Old vs uniform Material</title>'
        '<p>Old base / literal / learned M, then uniform-trained base / literal / learned M. '
        'Matched prompts, seeds and sampling; no Texture adapter or texture prompt.</p>'
        '<p>Quality and disentanglement require visual review.</p><a href="paired_blind_review.html">Paired blind review</a>'
        + "".join(f'<p><a href="{path}">{path}</a></p>' for path in sheets), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.old, args.new, args.output)
