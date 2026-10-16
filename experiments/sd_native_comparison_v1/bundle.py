"""Verify paired native runs and create SD1.4-left/SD1.5-right review sheets."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sd_native_comparison_v1.generate import (
    PROTOCOL, output_directory, read_protocol, rows, sha256, write_json,
)


def verify_pair(first, second):
    protocol = read_protocol()
    runs = []
    for directory, model in ((first, "sd14"), (second, "sd15")):
        manifest = json.loads((directory / "manifest.json").read_text())
        inventory = json.loads((directory / "model_inventory.json").read_text())
        records = [json.loads(line) for line in (directory / "ledger.jsonl").read_text().splitlines()]
        if (manifest.get("status") != "succeeded" or manifest.get("returncode") != 0
                or manifest.get("model") != model or manifest.get("adapters") is not False
                or manifest.get("protocol_sha256") != sha256(PROTOCOL)
                or manifest.get("generated_rows") != 20 or len(records) != 20
                or manifest["model_inventory_sha256"] != sha256(directory / "model_inventory.json")
                or not inventory.get("hashes_complete")
                or [r["id"] for r in records] != [r["id"] for r in rows(protocol, model)]):
            raise ValueError("native run is incomplete or its provenance differs")
        for record, expected in zip(records, rows(protocol, model)):
            if (any(record.get(key) != value for key, value in expected.items())
                    or record.get("status") != "succeeded" or not record.get("latent_sha256")
                    or type(record.get("safety_filtered")) is not bool
                    or sha256(directory / expected["image_path"]) != record["image_sha256"]):
                raise ValueError("native image ledger differs")
        runs.append((manifest, inventory, records))
    left, right = runs
    if (left[0]["git"] != right[0]["git"]
            or left[0]["script_sha256"] != right[0]["script_sha256"]
            or left[1]["safety_components"] != right[1]["safety_components"]
            or left[0]["safety_source"] != right[0]["safety_source"]):
        raise ValueError("the two arms used different source code or safety components")
    schedulers = [{k: v for k, v in run[0]["runtime"]["scheduler_config"].items()
                   if not k.startswith("_")} for run in runs]
    if schedulers[0] != schedulers[1]:
        raise ValueError("effective PNDM scheduler settings differ")
    for a, b in zip(left[2], right[2]):
        if any(a[key] != b[key] for key in ("pair_id", "prompt", "seed", "latent_sha256")):
            raise ValueError("prompt, seed or actual input noise differs between models")
    return left[2], right[2]


def make_sheets(first, second, output, left, right):
    from PIL import Image, ImageDraw
    protocol = read_protocol()
    summary = Image.new("RGB", (4 * 2 * 256, 28 + 5 * 282), "white")
    draw_summary = ImageDraw.Draw(summary)
    hashes = {}
    for object_index, obj in enumerate(protocol["objects"]):
        sheet = Image.new("RGB", (1024, 28 + 5 * 540), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"{obj}: SD1.4 (left) | SD1.5 (right)", fill="black")
        draw_summary.text((object_index * 512 + 4, 8), f"{obj}: SD1.4 | SD1.5", fill="black")
        for seed_index, seed in enumerate(protocol["sampling"]["seeds"]):
            for column, (root, records) in enumerate(((first, left), (second, right))):
                record = next(r for r in records if r["object"] == obj and r["seed"] == seed)
                with Image.open(root / record["image_path"]) as source:
                    image = source.convert("RGB")
                label = f"seed {seed}" + (" SAFETY FILTERED" if record["safety_filtered"] else "")
                x, y = column * 512, 28 + seed_index * 540
                sheet.paste(image, (x, y + 24))
                draw.text((x + 5, y + 5), label, fill="red" if record["safety_filtered"] else "black")
                sx, sy = object_index * 512 + column * 256, 28 + seed_index * 282
                summary.paste(image.resize((256, 256)), (sx, sy + 24))
                draw_summary.text((sx + 4, sy + 4), label, fill="black")
        path = output / (obj + ".jpg")
        sheet.save(path, quality=95)
        hashes[path.name] = sha256(path)
    summary.save(output / "overview.jpg", quality=95)
    hashes["overview.jpg"] = sha256(output / "overview.jpg")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sd14", type=Path, required=True)
    parser.add_argument("--sd15", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left, right = verify_pair(args.sd14, args.sd15)
    output = output_directory(args.output)
    output.mkdir(parents=True)
    sheets = make_sheets(args.sd14, args.sd15, output, left, right)
    write_json(output / "manifest.json", {
        "status": "succeeded", "sd14": str(args.sd14.resolve()), "sd15": str(args.sd15.resolve()),
        "protocol_sha256": sha256(PROTOCOL), "pair_count": len(left),
        "safety_filtered": {"sd14": sum(r["safety_filtered"] for r in left),
                            "sd15": sum(r["safety_filtered"] for r in right)},
        "sheets_sha256": sheets, "interpretation": "visual review pending; no aggregate quality claim",
    })
    print(output, flush=True)


if __name__ == "__main__":
    main()
