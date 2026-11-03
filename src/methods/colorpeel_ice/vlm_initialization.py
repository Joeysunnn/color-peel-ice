"""Single-image semantic initialization, independent of historical CLEVR labels.

Qwen supplies visible descriptions and ranked word candidates, not PBR numbers.
The diffusion tokenizer, rather than English word count, decides single-BPE fit.
Heavy model imports occur only during an explicit run.
"""

import hashlib
import json
from pathlib import Path
import re

from PIL import Image, ImageOps


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MAX_NEW_TOKENS = 512
TOKENS = {"subject": "<S*>", "color": "<C*>", "material": "<M*>", "texture": "<T*>"}
CONFIDENCES = {"high", "medium", "low", "unknown"}
REFLECTION_WORDS = {"glossy", "gloss", "rough", "roughness", "smooth", "matte", "shiny",
                    "polished", "reflective", "metallic", "specular", "lustrous"}
SUBSTRATE_WORDS = {"wood", "metal", "plastic", "rubber", "glass", "ceramic", "leather",
                   "fabric", "stone", "marble", "steel", "iron", "copper", "aluminum"}
PROMPT = """Inspect this single image. Identify the main foreground object and describe only
that same object's visible attributes. Ignore background walls, signs, lettering,
and numbers. Do not infer an object's identity from text printed in the image.
Return one JSON object only, with exactly these four keys and nested fields:
{
  "subject": {"category": "generic object category noun", "description": "brief visual identity evidence", "confidence": "high|medium|low|unknown", "candidates": ["word"]},
  "color": {"description": "dominant foreground color and visual evidence", "confidence": "high|medium|low|unknown", "candidates": ["word"]},
  "material": {"substrate": "possible substrate, or unknown", "visible_finish": "observed reflection or finish", "description": "brief visible finish evidence", "confidence": "high|medium|low|unknown", "candidates": ["word"]},
  "texture": {"description": "visible spatial surface pattern and evidence", "confidence": "high|medium|low|unknown", "candidates": ["word"]}
}
Use short English phrases, at most twelve words per description. Each candidates
list contains one to four ranked alternatives, each a SINGLE English word with
letters only: no spaces, hyphens, special tokens, or explanations. These words
initialize four independent learned concepts; do not combine attributes.
Subject candidates are object category nouns, not material or color words.
Subject description must exclude color, material, texture and background; category
is an ordinary English noun or short noun phrase using letters and spaces only.
Color candidates are ordinary color names describing the object, not background.
Material candidates are adjectives for the VISIBLE reflection or finish. Keep
possible underlying substrate separate: a painted object does not establish
bare-metal reflectance. Do not use substrate nouns as Material candidates.
Do not estimate numerical roughness, metallic, or other physical parameters.
Texture means a SPATIAL albedo pattern (marks, spots, stripes or grain), not
glossiness, smoothness, roughness, reflectivity, shape edges, or illumination.
Do not use glossy, rough, smooth, matte or similar reflection words as Texture
candidates. For no visible spatial pattern use plain or uniform; do not invent
grain. If an attribute cannot be determined, use confidence unknown and an empty
candidates list. Confidence is your qualitative assessment, not a calibrated
probability. Return JSON without Markdown or surrounding commentary."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_response(raw):
    response = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(response, dict) or set(response) != set(TOKENS):
        raise ValueError("Require exactly subject, color, material and texture")
    common = {"description", "confidence", "candidates"}
    for branch, value in response.items():
        extra = {"category"} if branch == "subject" else {"substrate", "visible_finish"} if branch == "material" else set()
        if not isinstance(value, dict) or set(value) != common | extra:
            raise ValueError(f"Unexpected {branch} response fields")
        for key in {"description"} | extra:
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError(f"{branch}.{key} must be a nonempty string")
        if not isinstance(value["confidence"], str) or value["confidence"] not in CONFIDENCES:
            raise ValueError(f"Invalid {branch} confidence")
        if branch == "subject" and re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+)*", value["category"]) is None:
            raise ValueError("Subject category must be an ordinary English noun phrase")
        candidates = value["candidates"]
        if (not isinstance(candidates, list) or len(candidates) > 4
                or any(not isinstance(word, str) for word in candidates)
                or (not candidates and value["confidence"] != "unknown")):
            raise ValueError(f"Invalid {branch} candidates")
        lowered = {word.strip().lower() for word in candidates}
        if branch == "texture" and lowered & REFLECTION_WORDS:
            raise ValueError("Texture candidates contain reflection/roughness words")
        if branch == "material" and lowered & SUBSTRATE_WORDS:
            raise ValueError("Material candidates contain substrate nouns, not visible finish adjectives")
    return response


def build_branches(response, tokenizer):
    from src.train.initializer_token_utils import select_initializer

    branches, failures = {}, []
    for branch, token in TOKENS.items():
        value = response[branch]
        word, token_id, audit = select_initializer(tokenizer, value["candidates"])
        branches[branch] = {**value, "token": token, "initializer": word,
                            "initializer_token_id": token_id, "candidate_audit": audit}
        if value["confidence"] == "unknown":
            failures.append(f"{branch}: unknown semantic confidence")
        if word is None:
            failures.append(f"{branch}: no valid single-BPE initializer")
    return branches, failures


def tokenizer_receipt(model_root):
    model_root = Path(model_root).resolve()
    folder = model_root / "tokenizer"
    files = {str(path.resolve()): sha256(path) for path in sorted(folder.rglob("*")) if path.is_file()}
    if not files:
        raise ValueError("Diffusion model has no local tokenizer files")
    return {"model_root": str(model_root), "files": files, "model_max_length": 77}


def model_receipt(model_dir):
    model_dir = Path(model_dir).resolve()
    if model_dir.name != REVISION:
        raise ValueError("Require the declared pinned Qwen snapshot directory")
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf8"))
    names = sorted(set(index["weight_map"].values()))
    if not names or any(Path(name).name != name for name in names):
        raise ValueError("Invalid Qwen weight index")
    weights = [{"path": str((model_dir / name).resolve()), "snapshot_file": name,
                "size_bytes": (model_dir / name).stat().st_size,
                "sha256": sha256(model_dir / name)} for name in names]
    metadata = {str(path.resolve()): sha256(path) for path in sorted(model_dir.iterdir())
                if path.is_file() and path.name not in names and path.suffix != ".safetensors"}
    return {"model_id": MODEL_ID, "revision": REVISION, "model_dir": str(model_dir),
            "metadata_files": metadata, "weights": weights,
            "weights_total_bytes": sum(row["size_bytes"] for row in weights),
            "weight_sha_method": "SHA256 computed from each complete file",
            "size_note": "File bytes are disk sizes, not GPU memory requirements."}


class QwenPredictor:
    def __init__(self, model_dir, device):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=torch.float16, attn_implementation="eager",
            low_cpu_mem_usage=True, local_files_only=True).to(device)
        self.model.eval()
        self.runtime = {"torch_version": torch.__version__, "transformers_version": transformers.__version__,
                        "dtype": "float16", "attention_implementation": "eager", "device": device,
                        "local_files_only": True, "batch_size": 1, "do_sample": False,
                        "max_new_tokens": MAX_NEW_TOKENS}
        if str(device).startswith("cuda"):
            self.runtime["torch_allocated_after_load_bytes"] = torch.cuda.memory_allocated(device)
            self.runtime["gpu_name"] = torch.cuda.get_device_name(device)

    def __call__(self, image):
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                   {"type": "text", "text": PROMPT}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.torch.float16)
        if str(self.device).startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
        tail = generated[:, inputs.input_ids.shape[1]:]
        self.runtime.update(input_tokens=int(inputs.input_ids.shape[1]), generated_tokens=int(tail.shape[1]),
                            reached_token_limit=bool(tail.shape[1] >= MAX_NEW_TOKENS))
        if str(self.device).startswith("cuda"):
            self.runtime["torch_peak_allocated_generation_bytes"] = self.torch.cuda.max_memory_allocated(self.device)
        return self.processor.batch_decode(tail, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0]


def run(image_path, model_dir, sd_model_dir, output, device="cuda:0"):
    image_path, model_dir, sd_model_dir, output = [Path(path).resolve() for path in
                                                  (image_path, model_dir, sd_model_dir, output)]
    repo = Path(__file__).resolve().parents[3]
    if output.exists() or any(output.is_relative_to(root) for root in (repo, model_dir, sd_model_dir)):
        raise ValueError("Require a fresh external output directory outside source/model directories")
    output.mkdir(parents=True)
    status = {"schema": "colorpeel_vlm_initializers/v1", "status": "started", "source": {"image": str(image_path)}}
    try:
        source_sha = sha256(image_path)
        status["source"]["image_sha256"] = source_sha
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
            status["source"].update(original_size=list(opened.size), input_size=list(image.size),
                                      conversion="EXIF transpose, RGB; no crop or mask")
        (output / "prompt.txt").write_text(PROMPT, encoding="utf8")
        receipt = model_receipt(model_dir)
        write_json(output / "model_receipt.json", receipt)
        tokenizer_info = tokenizer_receipt(sd_model_dir)
        predictor = QwenPredictor(model_dir, device)
        raw = predictor(image)
        (output / "raw_response.txt").write_text(raw, encoding="utf8")
        write_json(output / "generation_receipt.json", predictor.runtime)
        response = parse_response(raw)
        write_json(output / "response.json", response)
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(str(sd_model_dir), subfolder="tokenizer", local_files_only=True)
        if tokenizer.model_max_length != 77:
            raise ValueError("Require the SD2.1 CLIP tokenizer with model_max_length=77")
        branches, failures = build_branches(response, tokenizer)
        status.update(branches=branches, failures=failures)
        if failures:
            raise ValueError("; ".join(failures))
        locks = {str(image_path): source_sha, **receipt["metadata_files"], **tokenizer_info["files"]}
        for name in ("prompt.txt", "raw_response.txt", "response.json", "model_receipt.json", "generation_receipt.json"):
            locks[str(output / name)] = sha256(output / name)
        if any(sha256(path) != digest for path, digest in locks.items()):
            raise ValueError("Input or provenance changed during identification")
        manifest = {"schema": "colorpeel_vlm_initializers/v1", "status": "verified",
                    "source": status["source"], "vlm": {"model_id": MODEL_ID, "revision": REVISION,
                    **predictor.runtime, "model_receipt": str(output / "model_receipt.json")},
                    "tokenizer": tokenizer_info, "source_locks": locks, "branches": branches,
                    "subject_category": response["subject"]["category"], "response": response,
                    "scope": "Semantic initialization only; no new-image training dataset is claimed."}
        write_json(output / "initializers.json", manifest)
        from src.train.initializer_token_utils import load_initializer_manifest
        load_initializer_manifest(output / "initializers.json", tokenizer, sd_model_dir)
        status.update(status="verified", manifest_path=str(output / "initializers.json"))
        return output / "initializers.json"
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        if (output / "initializers.json").exists():
            failed = json.loads((output / "initializers.json").read_text(encoding="utf8"))
            failed["status"] = "failed"
            write_json(output / "initializers.json", failed)
        raise
    finally:
        write_json(output / "status.json", status)
