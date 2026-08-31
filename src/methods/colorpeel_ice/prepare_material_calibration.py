"""Prepare renderer-reference calibration and blinded material pair review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

from src.methods.colorpeel_ice.material_evaluation_protocol import read_protocol, validate_campaign


SHAPES = ("cube", "sphere", "cylinder")
COLORS = ("red", "cyan", "gray")
MATERIALS = ("metal", "rubber")
VIEWS = tuple(range(20))
PAIR_REVIEW_GENERATION_SEEDS = (42, 43)
PAIR_REVIEW_RANDOM_SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference_manifest(rows: Iterable[dict[str, Any]], reference_root: Path) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        shape, color, material, view = row.get("shape"), row.get("color"), row.get("material"), row.get("view_index")
        if shape not in SHAPES or color not in COLORS or material not in MATERIALS or view not in VIEWS:
            raise ValueError("reference row has an unsupported factor or view")
        relative = Path(row["image"])
        path = (reference_root / relative).resolve()
        recorded_hash = row.get("artifact_sha256", {}).get("image")
        if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
            raise ValueError("reference row is missing its image hash")
        items.append({
            "id": f"reference-{shape}-{color}-{material}-view-{view:02d}",
            "source": "accepted_multiview_render_v3_material",
            "shape": shape,
            "color": color,
            "material": material,
            "expected_shape": shape,
            "expected_color": color,
            "expected_material": material,
            "view_index": view,
            "render_seed": row.get("render_seed"),
            "cell_id": row.get("cell_id"),
            "image_path": str(path),
            "image_sha256": recorded_hash,
            "renderer_profile_id": row.get("renderer_profile_id"),
            "renderer_profile_sha256": row.get("renderer_profile_sha256"),
        })
    expected = {(shape, color, material, view) for shape in SHAPES for color in COLORS
                for material in MATERIALS for view in VIEWS}
    observed = {(row["shape"], row["color"], row["material"], row["view_index"]) for row in items}
    if len(items) != 360 or len({row["id"] for row in items}) != 360 or observed != expected:
        raise ValueError("reference calibration must contain the locked 3x3x2x20 grid")
    if len({row["image_path"] for row in items}) != 360 or len({row["image_sha256"] for row in items}) != 360:
        raise ValueError("reference calibration images and hashes must be unique")
    return sorted(items, key=lambda row: row["id"])


def validate_reference_files(items: Iterable[dict[str, Any]]) -> None:
    for item in items:
        path = Path(item["image_path"])
        if not path.is_file() or sha256_file(path) != item["image_sha256"]:
            raise ValueError(f"reference image missing or changed: {item['id']}")
        with Image.open(path) as image:
            image.load()
            if image.size != (512, 512) or image.mode != "RGB":
                raise ValueError(f"reference image is not 512x512 RGB: {item['id']}")


def material_pairs(rows: Iterable[dict[str, Any]], statuses: Iterable[dict[str, Any]],
                   protocol: dict[str, Any]) -> list[dict[str, Any]]:
    items = validate_campaign(rows, protocol)
    status_index = {row.get("id"): row for row in statuses}
    if len(status_index) != 3240 or set(status_index) != {row["id"] for row in items}:
        raise ValueError("generation status does not match the 3240-image campaign")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in items:
        key = (row["fold_id"], row["training_seed"], row["generation_seed"],
               row["expected_shape"], row["expected_color"])
        groups[key].append(row)
    if len(groups) != 1620:
        raise ValueError("expected 1620 material replacement pairs")
    output = []
    for key, pair in sorted(groups.items()):
        if len(pair) != 2 or {row["expected_material"] for row in pair} != set(MATERIALS):
            raise ValueError(f"invalid material pair: {key}")
        pair = sorted(pair, key=lambda row: row["expected_material"])
        for row in pair:
            status = status_index[row["id"]]
            path = Path(row["image_path"])
            if status.get("status") != "ok" or status.get("image_sha256") != sha256_file(path):
                raise ValueError(f"generated image missing or changed: {row['id']}")
            with Image.open(path) as image:
                image.load()
                if image.size != (512, 512) or image.mode != "RGB":
                    raise ValueError(f"generated image is not 512x512 RGB: {row['id']}")
        output.append({"group": key, "metal": next(row for row in pair if row["expected_material"] == "metal"),
                       "rubber": next(row for row in pair if row["expected_material"] == "rubber")})
    return output


def compose_pair(left: Path, right: Path, pair_id: str, output: Path) -> None:
    canvas = Image.new("RGB", (1040, 560), "white")
    draw = ImageDraw.Draw(canvas)
    with Image.open(left) as image:
        canvas.paste(image.convert("RGB"), (8, 40))
    with Image.open(right) as image:
        canvas.paste(image.convert("RGB"), (520, 40))
    draw.text((8, 10), pair_id, fill="black")
    draw.text((240, 10), "A", fill="black")
    draw.text((760, 10), "B", fill="black")
    canvas.save(output, quality=95)


def select_pair_review_pairs(pairs: list[dict[str, Any]]) -> list[tuple[dict[str, Any], bool]]:
    selected = [pair for pair in pairs if pair["group"][2] in PAIR_REVIEW_GENERATION_SEEDS]
    if len(selected) != 162:
        raise ValueError("balanced pair review must contain 162 pairs")
    rng = random.Random(PAIR_REVIEW_RANDOM_SEED)
    rng.shuffle(selected)
    return [(pair, rng.random() < 0.5) for pair in selected]


def create_pair_review(pairs: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    selected = select_pair_review_pairs(pairs)
    images_dir = output_dir / "pair_images"
    images_dir.mkdir()
    review, key_rows = [], []
    for index, (pair, metal_left) in enumerate(selected, 1):
        pair_id = f"material-pair-{index:04d}"
        left = pair["metal"] if metal_left else pair["rubber"]
        right = pair["rubber"] if metal_left else pair["metal"]
        image_path = images_dir / f"{pair_id}.jpg"
        compose_pair(Path(left["image_path"]), Path(right["image_path"]), pair_id, image_path)
        review.append({"review_order": index, "pair_id": pair_id, "pair_image": str(image_path.resolve()),
                       "more_metal_side_A_or_B": "", "confidence": "", "comment": ""})
        key_rows.append({"pair_id": pair_id, "expected_more_metal_side": "A" if metal_left else "B",
                         "fold_id": left["fold_id"], "training_seed": left["training_seed"],
                         "generation_seed": left["generation_seed"], "shape": left["expected_shape"],
                         "color": left["expected_color"], "left_id": left["id"],
                         "left_material": left["expected_material"], "right_id": right["id"],
                         "right_material": right["expected_material"]})
    write_csv(review, output_dir / "pair_review.csv")
    write_csv(key_rows, output_dir / "pair_key.csv")
    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir()
    for page in range(9):
        subset = review[page * 18:(page + 1) * 18]
        canvas = Image.new("RGB", (780, 900), "white")
        for slot, row in enumerate(subset):
            with Image.open(row["pair_image"]) as image:
                tile = image.convert("RGB")
                tile.thumbnail((256, 144))
            x, y = (slot % 3) * 260, (slot // 3) * 150
            canvas.paste(tile, (x + 2, y + 2))
        canvas.save(sheets_dir / f"page_{page + 1:02d}.jpg", quality=92)
    return {"pairs_total": len(pairs), "pairs_for_human_review": len(review),
            "generation_seeds": list(PAIR_REVIEW_GENERATION_SEEDS), "random_seed": PAIR_REVIEW_RANDOM_SEED,
            "contact_sheets": 9}


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    references = reference_manifest(read_jsonl(args.realized_views), args.reference_root.resolve())
    validate_reference_files(references)
    write_jsonl(references, output / "reference_manifest.jsonl")
    protocol = read_protocol(args.evaluation_protocol)
    pairs = material_pairs(read_jsonl(args.generation_manifest), read_jsonl(args.generation_status), protocol)
    pair_summary = create_pair_review(pairs, output)
    gate = {"status": "pending_human_pair_review", "human_review_is_primary": True,
            "qwen_reference_calibration_is_secondary": True,
            "required_field": "more_metal_side_A_or_B", "allowed_values": ["A", "B", "uncertain"]}
    (output / "human_gate_decision.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    result = {"status": "ready_for_qwen_reference_and_human_pair_review", "reference_images": 360,
              **pair_summary}
    (output / "preparation_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                                     encoding="utf-8")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realized-views", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--generation-status", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(prepare(parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
