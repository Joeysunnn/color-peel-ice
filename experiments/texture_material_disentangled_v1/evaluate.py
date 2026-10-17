"""Validate completed Phase 1 outputs and assemble oracle comparison pages."""

import argparse
from collections import Counter
import html
import itertools
import json
from pathlib import Path

from PIL import Image, ImageDraw

from experiments.texture_material_disentangled_v1 import data, infer


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def indexed(rows, expected, label):
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(expected):
        raise ValueError(f"{label} IDs are incomplete or duplicated")
    for identity, row in by_id.items():
        if any(row.get(key) != value for key, value in expected[identity].items()):
            raise ValueError(f"{label} condition changed: {identity}")
    return by_id


def validate(inference, renders):
    inference, renders = Path(inference).resolve(), Path(renders).resolve()
    protocol = data.read_protocol()
    provenance, summary = read_json(inference / "provenance.json"), read_json(inference / "summary.json")
    expected = {row["id"]: row for row in infer.build_rows(protocol)}
    if (provenance.get("status") != "succeeded" or summary.get("status") != "succeeded"
            or provenance.get("protocol_sha256") != data.protocol_sha256()
            or provenance.get("sampling") != protocol["inference"]
            or provenance.get("row_count") != len(expected)):
        raise ValueError("Inference is incomplete or its frozen protocol changed")
    indexed(read_rows(inference / "generation_manifest.jsonl"), expected, "Generation manifest")
    generated = indexed(read_rows(inference / "generation_status.jsonl"), expected, "Generation ledger")
    for row in generated.values():
        image = data._input_path(inference, row["image_path"])
        if row.get("status") not in ("ok", "safety_filtered") or data.sha256(image) != row.get("image_sha256"):
            raise ValueError(f"Invalid generation status or image hash: {row['id']}")
    filtered = sum(row["status"] == "safety_filtered" for row in generated.values())
    if (summary.get("expected_images") != len(expected) or summary.get("generated_images") != len(expected)
            or summary.get("missing_images") != 0 or summary.get("safety_filtered") != filtered
            or summary.get("by_method") != dict(Counter(row["method"] for row in generated.values()))):
        raise ValueError("Inference summary counts disagree with the complete ledger")
    oracle_manifest = read_json(renders / "manifest.json")
    if (oracle_manifest.get("status") != "complete"
            or oracle_manifest.get("protocol_sha256") != data.protocol_sha256()):
        raise ValueError("Oracle rendering is incomplete or its protocol changed")
    oracle_expected = {row["id"]: row for row in data._records(protocol)}
    oracle = indexed(oracle_manifest["records"], oracle_expected, "Oracle manifest")
    for row in oracle.values():
        material = protocol["materials"][row["material_id"]]
        if (any(row.get(key) != material[key] for key in ("roughness", "metallic"))
                or row.get("texture_projection") != "UV" or row.get("texture_colorspace") != "Non-Color"):
            raise ValueError("Oracle material or texture contract changed")
        for key in ("image", "mask"):
            if data.sha256(data._input_path(renders, row[key])) != row[key + "_sha256"]:
                raise ValueError(f"Oracle {key} hash changed: {row['id']}")
    if Counter(row["split"] for row in oracle.values()) != {"train": 64, "eval": 16}:
        raise ValueError("Oracle train/evaluation counts changed")
    return protocol, generated, oracle


def pages(output, inference, renders, protocol, generated, oracle):
    folder = output / "comparison_pages"
    folder.mkdir()
    pairs = list(itertools.product(infer.target_ids(protocol, "texture"), infer.target_ids(protocol, "material")))
    paths = []
    for method, context, seed in itertools.product(infer.METHODS, protocol["dataset"]["heldout_contexts"],
                                                  protocol["inference"]["seeds"]):
        sheet = Image.new("RGB", (5 * 224, 64 + 4 * 278), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"{method} / {context['id']} / diffusion seed {seed}", fill="black")
        draw.text((8, 23), "Oracle: UV grayscale x independent RGB; scalar R/M; no normal/bump.", fill="black")
        for column, title in enumerate(("PBR oracle", *protocol["inference"]["conditions"])):
            draw.text((column * 224 + 8, 44), title, fill="black")
        for line, (texture, material) in enumerate(pairs):
            reference = oracle[f"eval__{context['id']}__{texture}__{material}"]
            cards = [(data._input_path(renders, reference["image"]), False)]
            for condition in protocol["inference"]["conditions"]:
                row = generated[f"{method}__{texture}__{material}__{context['id']}__{condition}__seed{seed}"]
                cards.append((data._input_path(inference, row["image_path"]), row["status"] == "safety_filtered"))
            for column, (path, filtered) in enumerate(cards):
                x, y = column * 224 + 8, 64 + line * 278
                with Image.open(path) as image:
                    sheet.paste(image.convert("RGB").resize((208, 208)), (x, y))
                draw.text((x, y + 210), texture, fill="black")
                draw.text((x, y + 223), material, fill="black")
                draw.text((x, y + 236), f"Target R={reference['roughness']:g}, M={reference['metallic']:g}", fill="black")
                if filtered:
                    draw.text((x, y + 249), "SAFETY FILTERED", fill="red")
        relative = f"comparison_pages/{method}__{context['id']}__seed{seed}.jpg"
        sheet.save(output / relative, quality=90)
        paths.append(relative)
    return paths


def evaluate(inference, renders, output):
    inference, renders, output = (Path(path).resolve() for path in (inference, renders, output))
    if output.exists():
        raise FileExistsError(output)
    if any(output == source or source in output.parents for source in (inference, renders, infer.ROOT)):
        raise ValueError("Evaluation output must be outside immutable inputs and the source checkout")
    protocol, generated, oracle = validate(inference, renders)
    output.mkdir(parents=True)
    page_paths = pages(output, inference, renders, protocol, generated, oracle)
    groups = []
    for method, context, condition in itertools.product(infer.METHODS, protocol["dataset"]["heldout_contexts"],
                                                       protocol["inference"]["conditions"]):
        rows = [row for row in generated.values() if (row["method"], row["context"], row["condition"]) ==
                (method, context["id"], condition)]
        groups.append({"method": method, "context": context["id"], "condition": condition,
                       "generated": len(rows), "valid_observations": sum(row["status"] == "ok" for row in rows),
                       "safety_filtered": sum(row["status"] == "safety_filtered" for row in rows)})
    sources = [inference / name for name in ("provenance.json", "summary.json", "generation_manifest.jsonl",
                                            "generation_status.jsonl")] + [renders / "manifest.json"]
    report = {"status": "succeeded", "scientific_status": "pending_blinded_review",
              "training_steps": read_json(inference / "provenance.json").get("training_steps", 300),
              "protocol_sha256": data.protocol_sha256(), "generated_images": len(generated),
              "safety_filtered": sum(row["status"] == "safety_filtered" for row in generated.values()),
              "oracle_images": len(oracle), "oracle_evaluation_images": 16, "all_image_hashes_verified": True,
              "by_method_context_condition": groups, "comparison_pages": page_paths,
              "source_hashes": {str(path): data.sha256(path) for path in sources},
              "brdf_accuracy": None, "texture_accuracy": None, "concept_leakage_score": None,
              "phase2_started": False, "phase3_started": False}
    data.write_json(output / "report.json", report)
    links = "".join(f'<li><a href="{html.escape(path, quote=True)}">{html.escape(path)}</a></li>' for path in page_paths)
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Oracle comparison</title>'
        '<h1>Phase 1 oracle versus generated images</h1><p>Left: physical PBR reference. '
        'Remaining columns: literal, T-only, M-only, both. PBR and diffusion are different renderers; '
        'pixel equality is not expected. Filtered images are missing observations.</p><ul>' + links + '</ul>', encoding="utf-8")
    commit = read_json(inference / "provenance.json")["git"]["commit"]
    code = f"https://github.com/Joeysunnn/color-peel-ice/blob/{commit}/experiments/"
    (output / "REPORT.md").write_text(
        "# Phase 1 evaluation\n\nAll 256 generated images and 80 oracle render image/mask pairs passed their hash and condition checks. "
        f"Safety-filtered generations: {report['safety_filtered']}; these are missing observations.\n\n"
        f"The eight {report['training_steps']}-step snapshots test execution and learning, not successful disentanglement. "
        "Counterfactual balance and disjoint target-pair data establish the structural experimental design; "
        "learned semantic independence remains pending blinded review.\n\n"
        "The four contexts separate an unseen target pair under a seen context from individual held-out shape, color, "
        "and lighting changes. They do not test simultaneous mixed held-out factors. Oracle references show known "
        "PBR parameters; generated images have no measured BRDF. No pixel-similarity or execution count is a "
        "texture, reflectance, or leakage accuracy score.\n\n"
        "Use inference/blinded_review.html and blinded_ratings.csv to describe texture, material appearance, shape, "
        "color, lighting and background before joining blinded_key.json. Compare paired T/M swaps and literal controls. "
        "Do not infer physical R/M directly from visible gloss or uncalibrated VLM labels.\n\n"
        "Prior R/M-only training exhibited shape leakage; evaluator calibration found material label bias. "
        f"See the tracked [study README]({code}texture_material_disentangled_v1/README.md), "
        f"[previous material results]({code}natural_material_threeway_v1/RESULTS.md) and "
        f"[evaluator calibration]({code}clevr_subject_color_material_3x3x2/reports/08_material_evaluator_calibration.md). "
        "Phase 2 and Phase 3 remain unstarted pending scientific review.\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.inference, args.renders, args.output)
