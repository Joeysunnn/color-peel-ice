#!/usr/bin/env python3
"""Plan and strictly inject frozen D1 targets into neutral Rubber renders."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
import numpy as np
from src.methods.colorpeel_ice import neutral_lch_injection as injection

PLAN, CONTRACT, RESULTS, SUMMARY = "neutral_lch_plan.json", "neutral_lch_contract.json", "neutral_lch_results.json", "neutral_lch_analysis.json"

def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")

def plan(root: Path) -> dict[str, Any]:
    if root.exists() and any(root.iterdir()): raise injection.NeutralLchInjectionError("Output root must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    value = {"schema": "d1_neutral_lch_injection_plan/v1", "neutral_requests": injection.neutral_requests(),
             "injection_requests": injection.pilot_requests()}
    value["requests_sha256"] = canonical_sha256(value["injection_requests"])
    protocol = REPO_ROOT / injection.PROTOCOL_RELPATH
    contract = {"schema": "d1_neutral_lch_injection_contract/v1", "protocol_relative_path": injection.PROTOCOL_RELPATH,
                "protocol_sha256": file_sha256(protocol), "selection_canonical_sha256": selection_hash(),
                "plan_sha256": canonical_sha256(value)}
    _write(root / PLAN, value); _write(root / CONTRACT, contract); return value

def selection_hash() -> str:
    return injection.SELECTION_CANONICAL_SHA256

def _load_plan(root: Path, *, validate_injection_requests: bool = True) -> dict[str, Any]:
    value = json.loads((root / PLAN).read_text(encoding="utf-8"))
    if set(value) != {"schema", "neutral_requests", "injection_requests", "requests_sha256"}:
        raise injection.NeutralLchInjectionError("Plan fields differ")
    if value["schema"] != "d1_neutral_lch_injection_plan/v1" or value["neutral_requests"] != injection.neutral_requests():
        raise injection.NeutralLchInjectionError("Plan request identity differs")
    if validate_injection_requests and (value["injection_requests"] != injection.pilot_requests() or value["requests_sha256"] != canonical_sha256(injection.pilot_requests())):
        raise injection.NeutralLchInjectionError("Plan injection request identity differs")
    contract = json.loads((root / CONTRACT).read_text(encoding="utf-8"))
    protocol = REPO_ROOT / injection.PROTOCOL_RELPATH
    expected = {"schema": "d1_neutral_lch_injection_contract/v1", "protocol_relative_path": injection.PROTOCOL_RELPATH,
                "protocol_sha256": file_sha256(protocol), "selection_canonical_sha256": selection_hash(),
                "plan_sha256": canonical_sha256(value)}
    if contract != expected:
        raise injection.NeutralLchInjectionError("Plan provenance differs")
    return value

def render_neutral(root: Path, asset_root: Path) -> dict[str, Any]:
    """Render the three fixed neutral bases through the existing frozen Rubber adapter."""
    from scripts.methods.colorpeel_ice import render_d1_color_calibration_reduced_direct as direct
    from scripts.methods.colorpeel_ice import render_d1_color_calibration_preflight as shared
    require = injection.require
    require(shared.bpy is not None, "render-neutral requires Blender")
    plan_value = _load_plan(root, validate_injection_requests=False)
    output = root / "neutral"
    require(not output.exists(), "Neutral render output already exists")
    assets = {}
    for name, relative in direct.ASSETS.items():
        path = shared._inside(asset_root, relative, name)
        require(path.is_file(), f"Missing asset {name}")
        assets[name] = {"relative_path": relative, "sha256": file_sha256(path)}
    # The direct renderer's metadata validator is reused; only its color payload changes.
    linear = injection.lab_to_linear_rgb(np.array([injection.NEUTRAL_LAB]))[0].tolist()
    requests = []
    for base in plan_value["neutral_requests"]:
        requests.append({**base, "material": "Rubber", "target_a": 0.0, "target_b": 0.0,
                         "linear_rgb": linear, "socket_rgba": linear + [1.0], "preview_srgb_uint8": [0, 0, 0],
                         "stage": "neutral_lch_base"})
    contract = {"schema": "d1_neutral_lch_injection_render_contract/v1", "assets": assets}
    output.mkdir()
    records = []
    for request in requests:
        record = direct._render_one(output, request, contract, *(shared._inside(asset_root, assets[name]["relative_path"], name) for name in direct.ASSETS))
        records.append({"request_id": request["request_id"], "image_relative_path": record["image_relative_path"],
                        "image_sha256": record["image_sha256"], "mask_relative_path": record["mask_relative_path"],
                        "mask_sha256": record["mask_sha256"]})
    _write(output / "neutral_render_manifest.json", {"schema": "d1_neutral_lch_injection_neutral_manifest/v1", "records": records})
    return {"records": records}

def inject(root: Path, neutral_root: Path) -> dict[str, Any]:
    from PIL import Image
    value = _load_plan(root)
    if (root / RESULTS).exists(): raise injection.NeutralLchInjectionError("Injection results already exist")
    base = {row["view_index"]: row for row in value["neutral_requests"]}
    rows = []
    for request in value["injection_requests"]:
        source = base[request["view_index"]]
        directory = neutral_root / "renders" / source["request_id"]
        image_path, mask_path = directory / "image.png", directory / "mask.png"
        image = np.array(Image.open(image_path).convert("RGB")); mask = np.array(Image.open(mask_path).convert("L"))
        output, evidence = injection.inject_rgb(image, mask, request["target_a"], request["target_b"])
        row = {**request, **evidence, "neutral_image_relative_path": str(image_path.relative_to(neutral_root)),
               "neutral_mask_relative_path": str(mask_path.relative_to(neutral_root)),
               "neutral_image_sha256": file_sha256(image_path), "neutral_mask_sha256": file_sha256(mask_path)}
        if output is not None:
            out = root / "injected" / f"{request['request_id']}.png"; out.parent.mkdir(exist_ok=True)
            Image.fromarray(output, "RGB").save(out); row["image_relative_path"] = str(out.relative_to(root)); row["image_sha256"] = file_sha256(out)
        rows.append(row)
    _write(root / RESULTS, {"schema": "d1_neutral_lch_injection_results/v1", "rows": rows}); return {"rows": rows}

def analyze(root: Path, neutral_root: Path) -> dict[str, Any]:
    from PIL import Image
    rows = json.loads((root / RESULTS).read_text(encoding="utf-8"))["rows"]
    from scripts.methods.colorpeel_ice import render_d1_color_calibration_reduced_direct as direct
    for row in rows:
        if row["status"] == "injected":
            image = np.array(Image.open(root / row["image_relative_path"]).convert("RGB"))
            # The immutable neutral source mask is addressed by its view-specific request identity.
            mask_path = neutral_root / row["neutral_mask_relative_path"]
            if not mask_path.is_file():
                raise injection.NeutralLchInjectionError("Neutral mask is unavailable for analysis")
            measured = direct.measure_pixels(image.reshape(-1, 3), np.array(Image.open(mask_path).convert("L")).reshape(-1), row)
            row.update({key: measured[key] for key in ("e_ch", "median_L", "median_a", "median_b", "measurement_stats")})
    result = injection.summarize_measurements(rows); _write(root / SUMMARY, result); return result

def main(argv=None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("command", choices=["plan","render-neutral","inject","analyze"]); parser.add_argument("--output-root", required=True, type=Path); parser.add_argument("--neutral-root", type=Path); parser.add_argument("--asset-root", type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command == "plan": result = plan(args.output_root)
        elif args.command == "render-neutral": result = render_neutral(args.output_root, args.asset_root) if args.asset_root else (_ for _ in ()).throw(injection.NeutralLchInjectionError("--asset-root is required"))
        elif args.command == "inject": result = inject(args.output_root, args.neutral_root) if args.neutral_root else (_ for _ in ()).throw(injection.NeutralLchInjectionError("--neutral-root is required"))
        else: result = analyze(args.output_root, args.neutral_root) if args.neutral_root else (_ for _ in ()).throw(injection.NeutralLchInjectionError("--neutral-root is required"))
    except (OSError, ValueError, injection.NeutralLchInjectionError) as exc: parser.exit(2, f"neutral LCh injection aborted: {exc}\n")
    print(json.dumps(result, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
