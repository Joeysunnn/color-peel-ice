"""Stage rendered RGBA inputs and summarize SuperMat roughness calibration."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def cells(phase_dir):
    phase_dir = Path(phase_dir)
    for path in sorted(phase_dir.iterdir()):
        if path.is_dir() and (path / "metadata.json").is_file():
            yield path, json.loads((path / "metadata.json").read_text())


def stage(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    staged = []
    for cell, meta in cells(args.phase_dir):
        source = cell / "input_rgba.png"
        if sha(source) != meta["input_rgba_sha256"]:
            raise ValueError(f"RGBA hash differs from renderer metadata: {cell}")
        destination = out / f"{cell.name}.png"
        destination.write_bytes(source.read_bytes())
        staged.append({"cell_id": cell.name, "source": str(source.resolve()),
                       "source_sha256": sha(source), "staged_sha256": sha(destination)})
    expected = 21 if Path(args.phase_dir).name == "sweep" else 12
    if len(staged) != expected:
        raise ValueError(f"Expected {expected} render cells; got {len(staged)}")
    write_json(out.parent / f"{Path(args.phase_dir).name}_staging_manifest.json", staged)


def stats(map_path, mask):
    image = Image.open(map_path).convert("L")
    if image.size != mask.size:
        raise ValueError(f"Map/mask size mismatch: {map_path}")
    values = np.asarray(image, dtype=np.float64)[np.asarray(mask) >= 128] / 255.0
    if len(values) < 1000:
        raise ValueError(f"Insufficient interior pixels: {map_path}")
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return {"median": float(np.median(values)), "iqr": float(q75 - q25),
            "pixel_count": int(len(values)), "map_sha256": sha(map_path)}


def summarize(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    phase = Path(args.phase_dir).name
    target = json.loads(Path(args.source_material_json).read_text())["roughness"]
    rows = []
    for cell, meta in cells(args.phase_dir):
        supermat = Path(args.supermat_output) / cell.name
        mask_path = cell / "object_mask.png"
        if sha(mask_path) != meta["object_mask_sha256"]:
            raise ValueError(f"Mask hash differs from renderer metadata: {cell}")
        mask = Image.open(mask_path).convert("L").filter(ImageFilter.MinFilter(11))
        roughness = stats(supermat / "roughness.png", mask)
        metallic = stats(supermat / "metallic.png", mask)
        rows.append({
            "cell_id": cell.name, "shape": meta["shape"], "color": meta["color"],
            "lighting": meta["lighting"], "r_blender": meta["roughness"],
            "m_blender": meta["metallic"], "r_supermat": roughness["median"],
            "m_supermat": metallic["median"], "r_error": roughness["median"] - target,
            "r_iqr": roughness["iqr"], "m_iqr": metallic["iqr"],
            "interior_pixels": roughness["pixel_count"],
            "render_metadata_sha256": sha(cell / "metadata.json"),
            "rgba_sha256": meta["input_rgba_sha256"],
            "mask_sha256": meta["object_mask_sha256"],
            "r_map_sha256": roughness["map_sha256"],
            "m_map_sha256": metallic["map_sha256"],
        })
    expected = 21 if phase == "sweep" else 12
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} cells; got {len(rows)}")
    fields = list(rows[0])
    with (output / "per_cell.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {"schema_version": 1, "phase": phase, "target_roughness": target,
              "source_material_sha256": sha(args.source_material_json),
              "renderer_manifest_sha256": sha(Path(args.phase_dir) / "manifest.json"),
              "rows": rows}
    if phase == "sweep":
        mapping = []
        for r_input in sorted({row["r_blender"] for row in rows}):
            group = [row for row in rows if row["r_blender"] == r_input]
            if len(group) != 3:
                raise ValueError(f"Expected three lighting conditions at {r_input}")
            r_values = [row["r_supermat"] for row in group]
            mapping.append({"r_blender": r_input,
                            "r_supermat_median": statistics.median(r_values),
                            "r_supermat_min": min(r_values),
                            "r_supermat_max": max(r_values),
                            "m_supermat_median": statistics.median(
                                row["m_supermat"] for row in group)})
        selected = min(mapping, key=lambda item: (abs(item["r_supermat_median"] - target),
                                                  item["r_blender"]))
        report["empirical_mapping"] = mapping
        report["selected_grid_point"] = selected
    else:
        r_values = [row["r_supermat"] for row in rows]
        m_values = [row["m_supermat"] for row in rows]
        report["validation_summary"] = {
            "r_median": statistics.median(r_values), "r_min": min(r_values),
            "r_max": max(r_values), "r_range": max(r_values) - min(r_values),
            "median_absolute_r_error": statistics.median(abs(x - target) for x in r_values),
            "m_median": statistics.median(m_values), "m_max": max(m_values),
        }
    write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stage")
    p.add_argument("--phase-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=stage)
    p = sub.add_parser("summarize")
    p.add_argument("--phase-dir", required=True)
    p.add_argument("--supermat-output", required=True)
    p.add_argument("--source-material-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
