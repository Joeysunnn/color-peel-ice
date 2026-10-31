"""Link the pinned cached SD2.1 base and the existing common safety checker."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
REPO = "sd2-community/stable-diffusion-2-1-base"
REVISION = "4e63672c03103b6c636b8fb4119ba982469b2955"
CHECKER_REVISION = "133a221b8aa7292a167afc5127cb63fb5005638b"
SOURCE_DIR = Path("/home/r12user5/.cache/huggingface/hub/models--sd2-community--stable-diffusion-2-1-base/snapshots") / REVISION
CHECKER_DIR = Path("/home/r12user5/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots") / CHECKER_REVISION
MODEL_DIR = Path("/var/tmp/colorpeel-r12user5/assets/sd21_base_4e63672_common_checker")
COMPONENTS = ("unet", "text_encoder", "tokenizer", "vae", "scheduler", "safety_checker", "feature_extractor")
INITIALIZERS = ("spoon", "material", "texture")
ARCHITECTURE = {"cross_attention_dim": 1024, "text_hidden_size": 1024, "text_layers": 23,
                "prediction_type": "epsilon", "scheduler": "PNDMScheduler", "sample_size": 64,
                "use_linear_projection": True, "attention_head_dim": [5, 10, 20, 20]}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _text(value):
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initializer_ids(source):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(source), subfolder="tokenizer",
                                             local_files_only=True, use_fast=False)
    result = {}
    for word in INITIALIZERS:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1 or ids[0] in tokenizer.all_special_ids:
            raise ValueError(f"initializer must be one ordinary token: {word}")
        result[word] = int(ids[0])
    if tokenizer.model_max_length != 77:
        raise ValueError("expected the SD2.1 77-position tokenizer")
    return result


def _describe(source, checker, output):
    if source.name != REVISION or checker.name != CHECKER_REVISION:
        raise ValueError("source cache snapshot revision differs")
    index = _read(source / "model_index.json")
    expected_components = {"unet": ["diffusers", "UNet2DConditionModel"],
                           "vae": ["diffusers", "AutoencoderKL"],
                           "text_encoder": ["transformers", "CLIPTextModel"],
                           "tokenizer": ["transformers", "CLIPTokenizer"],
                           "scheduler": ["diffusers", "PNDMScheduler"]}
    if (index.get("_class_name") != "StableDiffusionPipeline"
            or any(index.get(key) != value for key, value in expected_components.items())):
        raise ValueError("source is not the pinned SD2.1 base pipeline")
    unet = _read(source / "unet/config.json")
    text_encoder = _read(source / "text_encoder/config.json")
    scheduler = _read(source / "scheduler/scheduler_config.json")
    architecture = {key: unet.get(key) for key in
                    ("cross_attention_dim", "sample_size", "use_linear_projection", "attention_head_dim")}
    architecture.update(text_hidden_size=text_encoder.get("hidden_size"),
                        text_layers=text_encoder.get("num_hidden_layers"),
                        prediction_type=scheduler.get("prediction_type"), scheduler=scheduler.get("_class_name"))
    if (architecture != ARCHITECTURE or text_encoder.get("architectures") != ["CLIPTextModel"]
            or text_encoder.get("max_position_embeddings") != 77
            or unet.get("block_out_channels") != [320, 640, 1280, 1280]
            or unet.get("in_channels") != 4 or unet.get("out_channels") != 4):
        raise ValueError("SD2.1 architecture or epsilon prediction configuration differs")
    required = {"unet": ("config.json",), "text_encoder": ("config.json",),
                "vae": ("config.json",), "scheduler": ("scheduler_config.json",),
                "tokenizer": ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"),
                "safety_checker": ("config.json",), "feature_extractor": ("preprocessor_config.json",)}
    weight_names = {"unet": ("diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"),
                    "vae": ("diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"),
                    "text_encoder": ("pytorch_model.bin", "model.safetensors"),
                    "safety_checker": ("pytorch_model.bin", "model.safetensors")}
    files, locks = {}, {str(source / "model_index.json"): _sha256(source / "model_index.json")}
    for component in COMPONENTS:
        folder = (checker if component in ("safety_checker", "feature_extractor") else source) / component
        for name in required[component]:
            if not (folder / name).is_file() or not (folder / name).stat().st_size:
                raise FileNotFoundError(folder / name)
        if component in weight_names and not any(
                (folder / name).is_file() and (folder / name).stat().st_size for name in weight_names[component]):
            raise FileNotFoundError(f"complete non-variant weights missing: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                digest = _sha256(path)
                files[(Path(component) / path.relative_to(folder)).as_posix()] = digest
                locks[str(path)] = digest
    initializer_ids = _initializer_ids(source)
    index.update(safety_checker=["stable_diffusion", "StableDiffusionSafetyChecker"],
                 feature_extractor=["transformers", "CLIPFeatureExtractor"], requires_safety_checker=True)
    files["model_index.json"] = hashlib.sha256(_text(index).encode("utf8")).hexdigest()
    receipt = {"model": {"repo": REPO, "revision": REVISION,
                         "official_repo": "stabilityai/stable-diffusion-2-1-base",
                         "source_note": "Official repository unavailable; use the pinned cached community mirror.",
                         "root": str(output), "source_root": str(source), "checker_root": str(checker),
                         "checker_repo": "CompVis/stable-diffusion-v1-4", "checker_revision": CHECKER_REVISION},
               "files": files, "source_locks": locks, "safety_checker_source": str(checker),
               "architecture": architecture, "initializer_ids": initializer_ids}
    return receipt, index


def verify_model(path=MODEL_DIR):
    """Verify every linked file against its immutable source receipt; return it."""
    path = Path(path).resolve()
    receipt = _read(path / "model_source.json")
    source, checker = (Path(receipt["model"][key]).resolve() for key in ("source_root", "checker_root"))
    expected, index = _describe(source, checker, path)
    if receipt != expected:
        raise ValueError("model receipt or source file locks differ")
    if {child.name for child in path.iterdir()} != set(COMPONENTS) | {"model_index.json", "model_source.json"}:
        raise ValueError("unexpected or missing model bundle entries")
    if (_read(path / "model_index.json") != index
            or _sha256(path / "model_index.json") != receipt["files"]["model_index.json"]):
        raise ValueError("model index changed or common safety checker is disabled")
    for component in COMPONENTS:
        origin = checker if component in ("safety_checker", "feature_extractor") else source
        if (path / component).resolve() != (origin / component).resolve():
            raise ValueError(f"model component is not linked to its locked source: {component}")
    return receipt


def prepare(output=MODEL_DIR, source=SOURCE_DIR, *, checker=CHECKER_DIR):
    """Create only a fresh external bundle; an existing exact bundle is read-only."""
    output, source, checker = (Path(value).resolve() for value in (output, source, checker))
    if (output.is_relative_to(ROOT) or output.is_relative_to(source) or source.is_relative_to(output)
            or output.is_relative_to(checker) or checker.is_relative_to(output)):
        raise ValueError("model output must be external and separate from the source caches")
    if output.exists():
        receipt = verify_model(output)
        if receipt["model"]["source_root"] != str(source) or receipt["model"]["checker_root"] != str(checker):
            raise ValueError("existing model bundle uses different source roots")
        return receipt
    receipt, index = _describe(source, checker, output)
    output.mkdir(parents=True)
    for component in COMPONENTS:
        origin = checker if component in ("safety_checker", "feature_extractor") else source
        (output / component).symlink_to(origin / component, target_is_directory=True)
    (output / "model_index.json").write_bytes(_text(index).encode("utf8"))
    (output / "model_source.json").write_bytes(_text(receipt).encode("utf8"))
    return receipt


def runtime_check(path=MODEL_DIR):
    """Check cached CPU config/tokenizer classes without allocating model weights."""
    from diffusers import PNDMScheduler
    from transformers import CLIPTextConfig

    receipt = verify_model(path)
    train_root = str(ROOT / "src/train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel

    path = str(Path(path).resolve())
    text_config = CLIPTextConfig.from_pretrained(path, subfolder="text_encoder", local_files_only=True)
    unet_config = UNet2DConditionModel.load_config(path, subfolder="unet", local_files_only=True)
    scheduler = PNDMScheduler.from_pretrained(path, subfolder="scheduler", local_files_only=True)
    arguments = inspect.signature(UNet2DConditionModel.__init__).parameters
    if (text_config.hidden_size != 1024 or text_config.num_hidden_layers != 23
            or scheduler.config.prediction_type != "epsilon"
            or any(key not in arguments or unet_config[key] != receipt["architecture"][key]
                   for key in ("cross_attention_dim", "sample_size", "use_linear_projection", "attention_head_dim"))):
        raise ValueError("installed runtime classes differ from the model architecture")
    return {"status": "verified", "device": "cpu", "weights_loaded": False,
            "architecture": receipt["architecture"], "initializer_ids": receipt["initializer_ids"],
            "text_conditioning": "23-layer CLIPTextModel last_hidden_state with final LayerNorm; no extra CLIP skip"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODEL_DIR)
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--checker", type=Path, default=CHECKER_DIR)
    parser.add_argument("--runtime-check", action="store_true")
    args = parser.parse_args()
    receipt = prepare(args.output, args.source, checker=args.checker)
    result = {"status": "prepared", "root": receipt["model"]["root"], "file_count": len(receipt["files"])}
    if args.runtime_check:
        result["runtime_check"] = runtime_check(args.output)
    print(_text(result), end="")


if __name__ == "__main__":
    main()
