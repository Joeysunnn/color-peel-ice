"""Freeze the existing five mailbox colors and selected historical MARBLE mask."""

import json
from pathlib import Path
import shutil

from PIL import Image

from experiments.sd15_token_local_kv_v1.contract import sha256


ROOT = Path(__file__).resolve().parents[2]
COLOR_MAPPING = (("pink", "red"), ("green", "green"), ("cyan", "cyan"),
                 ("blue", "blue"), ("purple", "magenta"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def rows(config):
    if ([(item["name"], item["source_color"]) for item in config["colors"]] != list(COLOR_MAPPING)
            or config["metallic"] != [-20, 20] or config["roughness"] != [-1, 0, 1]):
        raise ValueError("require the frozen five-color mapping and signed two-by-three M/R grid")
    result = []
    for color in config["colors"]:
        for metallic in config["metallic"]:
            for roughness in config["roughness"]:
                adjective = {-1: "smooth", 0: "moderately rough", 1: "rough"}[roughness]
                material = "metallic" if metallic < 0 else "non-metallic"
                result.append({"id": f"{color['name']}__M{metallic:+d}__R{roughness:+d}",
                    "color": color["name"], "controls": {"metallic": metallic, "roughness": roughness},
                    "material_state": "high" if metallic < 0 else "low",
                    "roughness_state": {-1: "low", 0: "medium", 1: "high"}[roughness],
                    "caption": f"a photo of <S*> mailbox in {color['name']} color with a {adjective} {material} surface",
                    "caption_status": "proposed conditioning label; actual appearance requires visual validation",
                    "training_eligible": False})
    return result


def _check_hashes(hashes, prefix=Path()):
    for name, expected in hashes.items():
        if sha256(prefix / name) != expected:
            raise ValueError(f"immutable source or prepared artifact changed: {name}")


def prepare(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config = read(config_path)
    planned = rows(config)
    original_root, mask_path = Path(config["original_root"]).resolve(), Path(config["mask"]).resolve()
    if (output.is_relative_to(ROOT) or ROOT.is_relative_to(output)
            or output.is_relative_to(original_root) or original_root.is_relative_to(output)
            or mask_path.is_relative_to(output) or config_path.is_relative_to(output)):
        raise ValueError("new output must be outside repository and existing source assets")
    locks = {str(config_path): sha256(config_path), str(Path(__file__).resolve()): sha256(__file__),
             str(mask_path): config["mask_sha256"]}
    originals = {}
    for color in config["colors"]:
        path = original_root / "category" / color["source_color"] / "images/image.png"
        originals[color["name"]] = path
        locks[str(path)] = color["image_sha256"]
    _check_hashes(locks)
    with Image.open(mask_path) as handle:
        if (handle.mode != "L" or handle.size != (512, 512)
                or {value for count, value in handle.getcolors(256) or []} != {0, 255}):
            raise ValueError("require the selected binary 512px L historical mask")
        mask = handle.convert("RGB").resize((1024, 1024), Image.Resampling.BICUBIC)
    images = {}
    for name, path in originals.items():
        with Image.open(path) as handle:
            if handle.mode != "RGB" or handle.size != (512, 512):
                raise ValueError(f"require frozen 512px RGB color source: {path}")
            images[name] = handle.resize((1024, 1024), Image.Resampling.BICUBIC)
    output.mkdir(parents=True)
    (output / "source").mkdir()
    shutil.copyfile(mask_path, output / "source/object_mask512.png")
    mask.save(output / "source/mask.png")
    sources = {}
    for name, image in images.items():
        directory = output / "source" / name
        directory.mkdir()
        shutil.copyfile(originals[name], directory / "original512.png")
        image.save(directory / "target.png")
        shutil.copyfile(directory / "target.png", directory / "reference.png")
        sources[name] = {"original": f"source/{name}/original512.png", "target": f"source/{name}/target.png",
                         "reference": f"source/{name}/reference.png", "mask": "source/mask.png"}
    plan = {"config": config, "rows": planned, "sources": sources, "source_hashes": locks}
    write(output / "config.json", config)
    write(output / "plan.json", plan)
    paths = [output / "config.json", output / "plan.json", *(output / "source").rglob("*.png")]
    write(output / "prepare_receipt.json", {"output_hashes": {
        path.relative_to(output).as_posix(): sha256(path) for path in paths}})
    return verify_prepared(output)


def verify_prepared(output):
    output = Path(output).resolve()
    receipt = read(output / "prepare_receipt.json")
    for relative in receipt["output_hashes"]:
        if Path(relative).is_absolute() or not (output / relative).resolve().is_relative_to(output):
            raise ValueError("prepared artifact path escapes output")
    _check_hashes(receipt["output_hashes"], output)
    plan = read(output / "plan.json")
    _check_hashes(plan["source_hashes"])
    if plan["config"] != read(output / "config.json") or plan["rows"] != rows(plan["config"]):
        raise ValueError("prepared config or grid differs")
    return plan
