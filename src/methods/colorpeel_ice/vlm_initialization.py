"""Single-image semantic initialization, independent of historical CLEVR labels.

Qwen supplies visible descriptions; Material uses controlled reflection classes, not PBR numbers.
The diffusion tokenizer, rather than English word count, decides single-BPE fit.
Heavy model imports occur only during an explicit run.
"""

import hashlib
import json
from pathlib import Path
import re

from PIL import Image, ImageOps

from .material_initialization import SCHEMA, select_material_initializer, validate_material

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MAX_NEW_TOKENS = 768
TOKENS = {"subject": "<S*>", "color": "<C*>", "material": "<M*>", "texture": "<T*>"}
CONFIDENCES = {"high", "medium", "low", "unknown"}
REFLECTION_WORDS = {"glossy", "gloss", "rough", "roughness", "smooth", "matte", "shiny",
                    "polished", "reflective", "metallic", "specular", "lustrous"}
PROMPT = """Inspect this single image. Identify the main foreground object and describe only
that same object's visible attributes. Ignore background walls, signs, lettering,
and numbers. Do not infer an object's identity from text printed in the image.
Return one JSON object only, with exactly these four keys and nested fields:
{
  "subject": {"category": "generic object category noun", "description": "brief visual identity evidence", "confidence": "high|medium|low|unknown", "candidates": ["word"]},
  "color": {"description": "dominant foreground color and visual evidence", "confidence": "high|medium|low|unknown", "candidates": ["word"]},
  "material": {
    "metallicity": {"label": "metallic|dielectric|mixed|uncertain", "confidence": "high|medium|low", "evidence": "visible reflection evidence"},
    "roughness": {"label": "low|medium|high|uncertain", "confidence": "high|medium|low", "evidence": "visible reflection evidence"}
  },
  "texture": {"description": "visible spatial surface pattern and evidence", "confidence": "high|medium|low|unknown", "candidates": ["word"]}
}
Use short English phrases, at most twelve words per description or evidence.
For Subject, Color and Texture, each candidates list contains one to four ranked
alternatives, each a SINGLE English word with
letters only: no spaces, hyphens, special tokens, or explanations. These words
initialize four independent learned concepts; do not combine attributes.
Subject candidates are object category nouns, not material or color words.
Subject description must exclude color, material, texture and background; category
is an ordinary English noun or short noun phrase using letters and spaces only.
Color candidates are ordinary color names describing the object, not background.
For Material return ONLY the two classifications above, never initializer words.
Judge the VISIBLE surface reflection, not what the object is manufactured from.
A coating over a metal object does not establish bare-metal reflectance.
Metallic means conductor-like reflection; dielectric means nonmetallic reflection;
mixed means visibly different surface types. Use uncertain when evidence is insufficient.
Bright or sharp highlights also occur on glossy dielectrics: they are roughness
evidence, not sufficient evidence of metallicity. Do not assign high-confidence
metallic from highlight sharpness, brightness, or object category alone. Look for
surface-wide conductor-like reflection versus a diffuse body with specular highlights;
if that distinction is not visible, use uncertain rather than guessing metallicity.
Roughness low means sharp highlights/reflections, medium means intermediate or
semi-gloss reflection, high means broad/diffuse reflections. It does not mean
macroscopic bumps, grooves, texture patterns, or object shape. Separate lighting
effects from surface properties as far as the image allows. Give each axis its own
confidence and short evidence; do not infer one axis from the other.
Do not estimate numerical roughness, metallic, or other physical parameters.
Texture means a SPATIAL albedo pattern (marks, spots, stripes or grain), not
glossiness, smoothness, roughness, reflectivity, shape edges, or illumination.
Do not use glossy, rough, smooth, matte or similar reflection words as Texture
candidates. For no visible spatial pattern use plain or uniform; do not invent
grain. If Subject, Color or Texture cannot be determined, use confidence unknown
and an empty candidates list. For an undetermined Material axis, use label
uncertain with confidence low and explain the missing evidence; never add candidates.
Confidence is your qualitative assessment, not a calibrated
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
        if branch == "material":
            validate_material(value)
            continue
        extra = {"category"} if branch == "subject" else set()
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
    return response


def build_branches(response, tokenizer):
    from src.train.initializer_token_utils import initializer_candidates, select_initializer

    branches, failures = {}, []
    for branch, token in TOKENS.items():
        value = response[branch]
        decision = select_material_initializer(value) if branch == "material" else None
        candidates = [decision['initializer']] if decision else initializer_candidates(branch, value)
        word, token_id, audit = select_initializer(tokenizer, candidates)
        branches[branch] = {**value, "token": token, "initializer": word,
                            "selection_candidates": candidates,
                            "initializer_token_id": token_id, "candidate_audit": audit}
        if decision:
            branches[branch]['selection_policy'] = decision
        elif value["confidence"] == "unknown":
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
    def __init__(self, model_dir, device, prompt=PROMPT):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch, self.device, self.prompt = torch, device, prompt
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
                                                   {"type": "text", "text": self.prompt}]}]
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


def run(image_path, model_dir, sd_model_dir, output, device="cuda:0", selected_branches=None,
        allow_partial=False):
    from src.train.initializer_token_utils import SELECTED_SCHEMA
    selected = list(TOKENS) if selected_branches is None else list(selected_branches)
    if not selected or len(set(selected)) != len(selected) or any(name not in TOKENS for name in selected):
        raise ValueError('select distinct known initializer branches')
    schema = SCHEMA if selected_branches is None else SELECTED_SCHEMA
    if allow_partial and selected_branches is None:
        raise ValueError('partial initialization requires explicit selected branches')
    image_path, model_dir, sd_model_dir, output = [Path(path).resolve() for path in
                                                  (image_path, model_dir, sd_model_dir, output)]
    repo = Path(__file__).resolve().parents[3]
    if output.exists() or any(output.is_relative_to(root) for root in (repo, model_dir, sd_model_dir)):
        raise ValueError("Require a fresh external output directory outside source/model directories")
    output.mkdir(parents=True)
    status = {"schema": schema, "status": "started", "source": {"image": str(image_path)}}
    try:
        source_sha = sha256(image_path)
        status["source"]["image_sha256"] = source_sha
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
            status["source"].update(original_size=list(opened.size), input_size=list(image.size),
                                      conversion="EXIF transpose, RGB; no crop or mask")
        prompt = PROMPT
        if selected_branches is not None:
            prompt += ('\nRequested learning branches: ' + ', '.join(selected) +
                '. Keep the complete four-key JSON schema. Embossed relief, grooves and bumps '
                'are geometry, not spatial albedo pattern. If there are no visible albedo marks, '
                'use plain or uniform for Texture. Provide multiple ranked semantic alternatives '
                'when possible, because an English word may split into several diffusion tokens.')
        (output / 'prompt.txt').write_text(prompt, encoding='utf8')
        receipt = model_receipt(model_dir)
        write_json(output / "model_receipt.json", receipt)
        tokenizer_info = tokenizer_receipt(sd_model_dir)
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(str(sd_model_dir), subfolder="tokenizer", local_files_only=True)
        if tokenizer.model_max_length != 77:
            raise ValueError("Require the SD2.1 CLIP tokenizer with model_max_length=77")
        predictor = QwenPredictor(model_dir, device)
        predictor.prompt = prompt
        attempt_files = []
        for attempt in range(1 if selected_branches is None else 2):
            (output / 'prompt.txt').write_text(predictor.prompt, encoding='utf8')
            raw = predictor(image)
            (output / 'raw_response.txt').write_text(raw, encoding='utf8')
            write_json(output / 'generation_receipt.json', predictor.runtime)
            if selected_branches is not None:
                folder = output / 'attempts' / str(attempt + 1)
                folder.mkdir(parents=True)
                for name in ('prompt.txt', 'raw_response.txt', 'generation_receipt.json'):
                    (folder / name).write_bytes((output / name).read_bytes())
                    attempt_files.append(folder / name)
            if predictor.runtime.get('reached_token_limit'):
                raise ValueError('Qwen response reached its token limit; raw output preserved')
            response = parse_response(raw)
            write_json(output / 'response.json', response)
            branches, all_failures = build_branches(response, tokenizer)
            failures = [reason for reason in all_failures if reason.split(':', 1)[0] in selected]
            if not failures or any('unknown semantic' in reason for reason in failures):
                break
            predictor.prompt = prompt + ('\nA prior response failed diffusion token validation. '
                'Reinspect the same image and return the complete JSON with alternative words '
                'for these rejected decisions; do not change visual evidence just to pass. '
                'If uncertain keep confidence unknown. Rejections: ' + json.dumps({
                    key: branches[key]['candidate_audit'] for key in selected
                    if branches[key]['initializer'] is None}))
        status.update(branches=branches, failures=failures)
        unavailable = {reason.split(':', 1)[0] for reason in failures}
        available = [name for name in selected if name not in unavailable]
        if failures and (not allow_partial or not available):
            raise ValueError("; ".join(failures))
        locks = {str(image_path): source_sha, **receipt["metadata_files"], **tokenizer_info["files"]}
        for name in ("prompt.txt", "raw_response.txt", "response.json", "model_receipt.json", "generation_receipt.json"):
            locks[str(output / name)] = sha256(output / name)
        locks.update({str(path): sha256(path) for path in attempt_files})
        if any(sha256(path) != digest for path, digest in locks.items()):
            raise ValueError("Input or provenance changed during identification")
        manifest = {"schema": schema, "status": "verified",
                    "source": status["source"], "vlm": {"model_id": MODEL_ID, "revision": REVISION,
                    **predictor.runtime, "model_receipt": str(output / "model_receipt.json")},
                    "tokenizer": tokenizer_info, "source_locks": locks, "branches": branches,
                    "subject_category": response["subject"]["category"], "response": response,
                    "scope": "Semantic initialization only; no new-image training dataset is claimed."}
        if selected_branches is not None:
            manifest['selected_branches'] = available
            manifest['requested_branches'] = selected
            manifest['unavailable_branches'] = failures
            manifest['selection_attempts'] = len(attempt_files) // 3
        write_json(output / "initializers.json", manifest)
        from src.train.initializer_token_utils import load_initializer_manifest
        load_initializer_manifest(output / "initializers.json", tokenizer, sd_model_dir)
        status.update(status="partial" if failures else "verified", manifest_path=str(output / "initializers.json"))
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
