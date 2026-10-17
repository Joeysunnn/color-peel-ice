"""Independent oracle textures, balanced render plan, and verified training staging."""

import argparse
import collections
import hashlib
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.train.instance_mask_utils import load_latent_instance_mask, pair_instance_images_and_masks


PROTOCOL = Path(__file__).with_name("protocol.json")


def read_protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf8"))


def protocol_sha256():
    payload = json.dumps(read_protocol(), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf8")
    return hashlib.sha256(payload).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")


def extract_texture(albedo, valid_mask=None):
    """Return a linear-luminance, Non-Color L image and extraction statistics."""
    config = read_protocol()["texture_extraction"]
    image = albedo if isinstance(albedo, Image.Image) else Image.open(albedo)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    gray = linear @ np.array([0.2126, 0.7152, 0.0722])
    if valid_mask is None:
        valid = np.ones(gray.shape, dtype=bool)
    else:
        mask = valid_mask if isinstance(valid_mask, Image.Image) else Image.open(valid_mask)
        valid = np.asarray(mask.convert("L")) >= 128
        if valid.shape != gray.shape:
            raise ValueError("Albedo and valid mask dimensions differ")
    if valid.sum() < 2:
        raise ValueError("At least two valid albedo pixels are required")
    lower, upper = np.quantile(gray[valid], config["quantiles"])
    if upper - lower <= config["constant_epsilon"]:
        raise ValueError("Near-constant albedo has no reliable spatial texture")
    low, high = config["output_range"]
    normalized = low + np.clip((gray - lower) / (upper - lower), 0, 1) * (high - low)
    # Outside-mask pixels are neutral; they cannot enter quantiles or become texture evidence.
    normalized[~valid] = (low + high) / 2
    result = Image.fromarray(np.rint(normalized * 255).astype("uint8"))
    return result, {
        "method": config["method"], "quantiles": config["quantiles"],
        "lower_luminance": float(lower), "upper_luminance": float(upper),
        "valid_pixels": int(valid.sum()), "output_range": config["output_range"],
        "constant_epsilon": config["constant_epsilon"],
        "color_space": "linear luminance encoded directly as Non-Color grayscale",
        "outside_mask": "neutral midpoint; not used for quantiles",
    }


def oracle_albedo(pattern, size):
    y, x = np.indices((size, size))
    cell = size // 4
    if pattern == "vertical_stripes":
        bright = (x // (cell // 2)) % 2 == 0
    elif pattern == "horizontal_stripes":
        bright = (y // (cell // 2)) % 2 == 0
    elif pattern == "checkerboard":
        bright = (x // cell + y // cell) % 2 == 0
    elif pattern == "dots":
        bright = ((x % cell - (cell - 1) / 2) ** 2 +
                  (y % cell - (cell - 1) / 2) ** 2) <= (cell * 0.4) ** 2
    else:
        raise ValueError(f"Unknown oracle pattern: {pattern}")
    return Image.fromarray(np.where(bright, 204, 51).astype("uint8")).convert("RGB")


def _records(protocol, texture_paths=None):
    texture_paths = texture_paths or {}
    dataset = protocol["dataset"]
    rows = []
    for branch in ("texture", "material"):
        target_kind, nuisance_kind = (("textures", "materials") if branch == "texture"
                                     else ("materials", "textures"))
        targets = [key for key, value in protocol[target_kind].items() if value["role"] == "target"]
        nuisances = [key for key, value in protocol[nuisance_kind].items() if value["role"] == "nuisance"]
        contexts = list(itertools.product(dataset["train_shapes"], dataset["train_colors"],
                                          dataset["train_lights"], nuisances))
        for target in targets:
            for index, (shape, color, light, nuisance) in enumerate(contexts):
                texture, material = ((target, nuisance) if branch == "texture" else (nuisance, target))
                rows.append({
                    "id": f"train__{branch}__{target}__{shape}__{color}__{light}__{nuisance}",
                    "split": "train", "branch": branch, "concept_id": target,
                    "texture_id": texture, "material_id": material,
                    "shape": shape, "color": color, "light": light,
                    "seed": protocol["seed"] + index,
                })
    target_textures = [key for key, value in protocol["textures"].items() if value["role"] == "target"]
    target_materials = [key for key, value in protocol["materials"].items() if value["role"] == "target"]
    for index, context in enumerate(dataset["heldout_contexts"]):
        for texture, material in itertools.product(target_textures, target_materials):
            rows.append({
                "id": f'eval__{context["id"]}__{texture}__{material}',
                "split": "eval", "branch": "composition", "concept_id": "composition",
                "context_id": context["id"], "texture_id": texture, "material_id": material,
                "shape": context["shape"], "color": context["color"], "light": context["light"],
                "seed": protocol["seed"] + 1000 + index,
            })
    if texture_paths:
        for row in rows:
            texture = Path(texture_paths[row["texture_id"]])
            row.update(texture_path=str(texture.resolve()), texture_sha256=sha256(texture))
    return rows


def make_plan(root):
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(root)
    protocol = read_protocol()
    root.mkdir(parents=True)
    texture_root = root / "textures"
    texture_root.mkdir()
    textures = {}
    extraction = {}
    for name, spec in protocol["textures"].items():
        albedo = oracle_albedo(spec["pattern"], protocol["texture_extraction"]["size"])
        texture, statistics = extract_texture(albedo)
        source = texture_root / f"{name}__albedo.png"
        output = texture_root / f"{name}.png"
        albedo.save(source)
        texture.save(output)
        textures[name] = output
        extraction[name] = {**statistics, "source_sha256": sha256(source),
                            "texture_sha256": sha256(output), "pattern": spec["pattern"]}
    rows = _records(protocol, textures)
    result = {"status": "complete", "protocol_sha256": protocol_sha256(),
              "train_images": sum(row["split"] == "train" for row in rows),
              "eval_images": sum(row["split"] == "eval" for row in rows),
              "texture_extraction": extraction, "records": rows}
    write_json(root / "plan.json", result)
    return result


def caption(row, protocol=None):
    protocol = protocol or read_protocol()
    texture = ("<T*>" if row["branch"] == "texture"
               else protocol["textures"][row["texture_id"]]["literal"])
    material = ("<M*>" if row["branch"] == "material"
                else protocol["materials"][row["material_id"]]["literal"])
    light = protocol["dataset"]["light_phrases"][row["light"]]
    return (f'a photo of a {row["color"]} {row["shape"]} with {texture} texture '
            f"made of {material} under {light}")


def _input_path(root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Render path escapes its root")
    return path


def stage(renderroot, output):
    renderroot, output = Path(renderroot).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = renderroot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    protocol = read_protocol()
    if manifest.get("status") != "complete" or manifest.get("protocol_sha256") != protocol_sha256():
        raise ValueError("Incomplete render manifest or changed protocol")
    rows = manifest["records"]
    expected = {row["id"]: row for row in _records(protocol)}
    by_id = {row["id"]: row for row in rows}
    if len(rows) != len(expected) or len(by_id) != len(rows) or set(by_id) != set(expected):
        raise ValueError("Render records do not match the exact 64-train/16-eval factorial")
    validated = {}
    nuisance_counts = collections.defaultdict(collections.Counter)
    mask_pairs = collections.defaultdict(list)
    checked_textures = set()
    for row in rows:
        if any(row.get(key) != value for key, value in expected[row["id"]].items()):
            raise ValueError(f'Changed factorial condition: {row["id"]}')
        material = protocol["materials"][row["material_id"]]
        for key in ("roughness", "metallic"):
            if row.get(key) != material[key]:
                raise ValueError(f'Changed material parameter: {row["id"]}/{key}')
        if row.get("texture_projection") != "UV":
            raise ValueError("Texture must use UV coordinates")
        if row.get("uv_repeats") != protocol["texture_extraction"]["uv_repeats"]:
            raise ValueError("Changed fixed UV repeats")
        if (row.get("texture_colorspace") != "Non-Color" or row.get("normal_bump_displacement") is not False
                or row.get("base_color_linear_rgb") != protocol["dataset"]["colors_linear_rgb"][row["color"]]
                or not row.get("uv_mapping") or len(row.get("uv_sha256", "")) != 64):
            raise ValueError("Missing or incompatible shader/UV evidence")
        metadata = _input_path(renderroot, row["metadata"])
        if sha256(metadata) != row["metadata_sha256"]:
            raise ValueError("Changed render metadata")
        recorded = json.loads(metadata.read_text(encoding="utf8"))
        if recorded != {key: value for key, value in row.items() if key not in ("metadata", "metadata_sha256")}:
            raise ValueError("Render metadata disagrees with manifest")
        texture = Path(row["texture_path"])
        if sha256(texture) != row["texture_sha256"]:
            raise ValueError(f'Changed texture asset: {row["id"]}')
        texture_key = (row["texture_id"], row["texture_sha256"])
        if texture_key not in checked_textures:
            expected_texture, _ = extract_texture(oracle_albedo(
                protocol["textures"][row["texture_id"]]["pattern"], protocol["texture_extraction"]["size"]))
            with Image.open(texture) as opened:
                if opened.mode != "L" or not np.array_equal(np.asarray(opened), np.asarray(expected_texture)):
                    raise ValueError("Texture asset does not match the declared oracle pattern")
            checked_textures.add(texture_key)
        image, mask = (_input_path(renderroot, row[key]) for key in ("image", "mask"))
        if sha256(image) != row["image_sha256"] or sha256(mask) != row["mask_sha256"]:
            raise ValueError(f'Changed render image/mask: {row["id"]}')
        with Image.open(image) as opened, Image.open(mask) as opened_mask:
            size = protocol["renderer"]["resolution"]
            if opened.size != (size, size) or opened_mask.size != opened.size:
                raise ValueError(f'Wrong render dimensions: {row["id"]}')
            pixels = np.asarray(opened_mask.convert("L")) >= 128
            if not pixels.any():
                raise ValueError(f'Empty object mask: {row["id"]}')
            mask_pixels = np.asarray(opened_mask.convert("L")).tobytes()
        validated[row["id"]] = (image, mask)
        if row["split"] == "train":
            nuisance = row["material_id"] if row["branch"] == "texture" else row["texture_id"]
            nuisance_counts[(row["branch"], row["concept_id"])][(
                row["shape"], row["color"], row["light"], nuisance, row["seed"])] += 1
            mask_pairs[(row["branch"], row["shape"], row["color"], row["light"], nuisance,
                        row["seed"])].append(mask_pixels)
    for branch in ("texture", "material"):
        counts = [value for (kind, _), value in nuisance_counts.items() if kind == branch]
        if len(counts) != 2 or counts[0] != counts[1]:
            raise ValueError("Target concepts have different nuisance distributions")
    if any(any(mask != masks[0] for mask in masks) for masks in mask_pairs.values()):
        raise ValueError("Target swaps changed paired foreground masks")
    output.mkdir(parents=True)
    contacts = Image.new("RGB", (160 * 8, 190 * 8), "white")
    draw = ImageDraw.Draw(contacts)
    datasets = {}
    for concept_index, ((branch, concept), counts) in enumerate(sorted(nuisance_counts.items())):
        root = output / concept
        (root / "images").mkdir(parents=True)
        (root / "masks").mkdir()
        assets, concepts = [], []
        chosen = sorted((row for row in rows if row["split"] == "train" and
                         row["concept_id"] == concept), key=lambda row: row["id"])
        for index, row in enumerate(chosen):
            # One concept group per exact caption keeps every nuisance cell equally sampled.
            image_dir, mask_dir = root / "images" / row["id"], root / "masks" / row["id"]
            image_dir.mkdir()
            mask_dir.mkdir()
            image, mask = image_dir / "image.png", mask_dir / "image.png"
            shutil.copyfile(validated[row["id"]][0], image)
            with Image.open(validated[row["id"]][1]) as source_mask:
                binary = (np.asarray(source_mask.convert("L")) >= 128).astype("uint8") * 255
                Image.fromarray(binary).save(mask)
            if pair_instance_images_and_masks(image_dir, mask_dir) != [(image, mask)]:
                raise ValueError("Staged image and mask cannot be paired by the training loader")
            load_latent_instance_mask(mask, protocol["renderer"]["resolution"],
                                      protocol["renderer"]["resolution"] // 8)
            prompt = caption(row, protocol)
            concepts.append({"instance_prompt": [prompt], "instance_data_dir": str(image_dir),
                             "instance_mask_dir": str(mask_dir)})
            assets.append({**row, "prompt": prompt, "image": str(image), "mask": str(mask),
                           "source_image_sha256": row["image_sha256"],
                           "source_mask_sha256": row["mask_sha256"],
                           "image_sha256": sha256(image), "mask_sha256": sha256(mask)})
            with Image.open(image) as opened:
                thumb = opened.convert("RGB").resize((160, 160))
            slot = concept_index * len(chosen) + index
            x, y = (slot % 8) * 160, (slot // 8) * 190
            contacts.paste(thumb, (x, y))
            draw.text((x + 2, y + 161), f"{concept[:23]}\n{row['shape']} {row['color']}", fill="black")
        write_json(root / "concepts.json", concepts)
        ledger = root / "training_assets_manifest.jsonl"
        ledger.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in assets), encoding="utf8")
        receipt = {"status": "verified", "phase": 1, "branch": branch, "role": branch,
                   "token": "<T*>" if branch == "texture" else "<M*>", "concept_id": concept,
                   "sample_count": len(assets), "image_count": len(assets), "protocol_sha256": protocol_sha256(),
                   "concepts_sha256": sha256(root / "concepts.json"),
                   "training_assets_manifest_sha256": sha256(ledger),
                   "source_render_manifest_sha256": sha256(manifest_path)}
        receipt["mask_staging"] = "threshold raw renderer L mask >=128 to strict 0/255; matching image stem"
        write_json(root / "dataset_receipt.json", receipt)
        datasets[concept] = {"path": str(root), "dataset_receipt_sha256": sha256(root / "dataset_receipt.json"), **receipt}
    contacts.save(output / "training_contact_sheet.jpg", quality=90)
    audit = {"status": "complete", "protocol_sha256": protocol_sha256(),
             "source_render_manifest_sha256": sha256(manifest_path),
             "train_images": 64, "eval_images": 16, "datasets": datasets,
             "balanced_nuisance_distributions": True, "paired_target_masks_identical": True,
             "learned_texture_material_pairs_in_training": 0,
             "interpretation": "Dataset/control integrity only; not a learned disentanglement result"}
    write_json(output / "audit.json", audit)
    return audit


def verify_staging(root):
    root = Path(root).resolve()
    audit = json.loads((root / "audit.json").read_text(encoding="utf8"))
    if audit.get("status") != "complete" or audit.get("protocol_sha256") != protocol_sha256():
        raise ValueError("Staging audit is incomplete or belongs to another protocol")
    expected = collections.defaultdict(dict)
    for row in _records(read_protocol()):
        if row["split"] == "train":
            expected[row["concept_id"]][row["id"]] = row
    if set(audit["datasets"]) != set(expected):
        raise ValueError("Staging concept membership changed")
    for concept, conditions in expected.items():
        folder = root / concept
        receipt_path = folder / "dataset_receipt.json"
        if sha256(receipt_path) != audit["datasets"][concept]["dataset_receipt_sha256"]:
            raise ValueError(f"Changed dataset receipt: {concept}")
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        concepts_path, ledger = folder / "concepts.json", folder / "training_assets_manifest.jsonl"
        if (receipt.get("status") != "verified" or receipt.get("protocol_sha256") != protocol_sha256()
                or receipt.get("image_count") != len(conditions)
                or receipt["concepts_sha256"] != sha256(concepts_path)
                or receipt["training_assets_manifest_sha256"] != sha256(ledger)):
            raise ValueError(f"Changed staging metadata: {concept}")
        rows = [json.loads(line) for line in ledger.read_text(encoding="utf8").splitlines()]
        if len(rows) != len(conditions) or {row["id"] for row in rows} != set(conditions):
            raise ValueError(f"Changed staging sample membership: {concept}")
        expected_concepts, expected_images, expected_masks = [], set(), set()
        for row in rows:
            if any(row.get(key) != value for key, value in conditions[row["id"]].items()):
                raise ValueError(f"Changed staging condition: {row['id']}")
            image = folder / "images" / row["id"] / "image.png"
            mask = folder / "masks" / row["id"] / "image.png"
            if (Path(row["image"]) != image or Path(row["mask"]) != mask
                    or sha256(image) != row["image_sha256"] or sha256(mask) != row["mask_sha256"]
                    or row["prompt"] != caption(row)):
                raise ValueError(f"Changed staged image, mask or caption: {row['id']}")
            if pair_instance_images_and_masks(image.parent, mask.parent) != [(image, mask)]:
                raise ValueError(f"Training-loader image/mask pairing failed: {row['id']}")
            size = read_protocol()["renderer"]["resolution"]
            load_latent_instance_mask(mask, size, size // 8)
            expected_images.add(image)
            expected_masks.add(mask)
            expected_concepts.append({"instance_prompt": [caption(row)],
                                      "instance_data_dir": str(image.parent),
                                      "instance_mask_dir": str(mask.parent)})
        if json.loads(concepts_path.read_text(encoding="utf8")) != expected_concepts:
            raise ValueError(f"Changed concept sampling groups: {concept}")
        for name, members in (("images", expected_images), ("masks", expected_masks)):
            if {path for path in (folder / name).rglob("*") if path.is_file()} != members:
                raise ValueError(f"Unexpected staged files: {concept}/{name}")
    return audit


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    staging = sub.add_parser("stage")
    staging.add_argument("--renders", type=Path, required=True)
    staging.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = make_plan(args.output) if args.command == "plan" else stage(args.renders, args.output)
    print(json.dumps({key: value for key, value in result.items() if key not in ("records", "texture_extraction")}, indent=2))


if __name__ == "__main__":
    main()
