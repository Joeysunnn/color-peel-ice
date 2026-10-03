"""Get categorical surface priors from Qwen3-VL without asking for PBR values."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image


PROMPT = """Look only at the masked target: {target}. Describe the visible surface,
not the hidden substrate. Return JSON only with exactly these keys:
{{"surface_family":"coated_or_painted|bare_metal|plastic|uncertain",
"gloss":"matte|semi_gloss|glossy|uncertain",
"opacity":"opaque|translucent|uncertain","evidence":"brief visual cue"}}.
Choose one allowed value per field. Do not estimate numerical roughness or
metallic, and do not use color names in the evidence. Use uncertain if needed."""
ALLOWED = {
    "surface_family": {"coated_or_painted", "bare_metal", "plastic", "uncertain"},
    "gloss": {"matte", "semi_gloss", "glossy", "uncertain"},
    "opacity": {"opaque", "translucent", "uncertain"},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def masked_crop(image_path, mask_path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if image.size != mask.size or mask.getbbox() is None:
        raise ValueError(f"Invalid image/mask pair: {image_path}, {mask_path}")
    left, top, right, bottom = mask.getbbox()
    padding = max(1, round(max(right - left, bottom - top) * 0.1))
    box = (max(0, left - padding), max(0, top - padding),
           min(image.width, right + padding), min(image.height, bottom + padding))
    neutral = Image.new("RGB", image.size, (127, 127, 127))
    neutral.paste(image, mask=mask)
    return neutral.crop(box)


def parse_response(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON object in VLM output: {raw}")
    result = json.loads(text[start:end + 1])
    if set(result) != set(ALLOWED) | {"evidence"}:
        raise ValueError(f"Wrong VLM output keys: {raw}")
    for field, choices in ALLOWED.items():
        if result[field] not in choices:
            raise ValueError(f"Invalid {field}: {raw}")
    if not isinstance(result["evidence"], str):
        raise ValueError(f"Invalid evidence: {raw}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.config, args.project_root = args.config.resolve(), args.project_root.resolve()
    args.model_path, args.output_root = args.model_path.resolve(), args.output_root.resolve()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    config = json.loads(args.config.read_text())
    args.output_root.mkdir(parents=True)

    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    results = []
    for sample in config["samples"]:
        folder = args.output_root / sample["id"]
        folder.mkdir()
        image_path = args.project_root / sample["image"]
        mask_path = args.project_root / sample["vlm_mask"]
        crop_path = folder / "masked_crop.png"
        masked_crop(image_path, mask_path).save(crop_path)
        prompt = PROMPT.format(target=sample["target"])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(crop_path)},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False,
                                    max_new_tokens=config["max_new_tokens"])
        raw = processor.batch_decode(
            [ids[len(input_ids):] for input_ids, ids in zip(inputs.input_ids, output)],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        (folder / "raw_response.txt").write_text(raw + "\n")
        prior = parse_response(raw)
        record = {"sample": sample, "image_sha256": sha(image_path),
                  "vlm_mask_sha256": sha(mask_path), "crop_sha256": sha(crop_path),
                  "prompt": prompt, "raw_response": raw, "prior": prior}
        (folder / "prior.json").write_text(json.dumps(record, indent=2) + "\n")
        results.append({"id": sample["id"], "prior": prior})

    manifest = {"status": "complete", "model_id": config["model_id"],
                "model_snapshot": args.model_path.name,
                "model_config_sha256": sha(args.model_path / "config.json"),
                "config_sha256": sha(args.config), "script_sha256": sha(__file__),
                "project_git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=args.project_root, text=True).strip(),
                "seed": config["seed"], "dtype": "float16", "do_sample": False,
                "max_new_tokens": config["max_new_tokens"], "results": results}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
