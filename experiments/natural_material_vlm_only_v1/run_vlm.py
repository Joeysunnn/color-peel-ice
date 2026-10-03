"""One masked Qwen3-VL semantic prediction per natural material sample."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image


PROMPT = """Look only at the masked target: {target}. Classify its *visible surface appearance*.
Choose exactly one label: painted_or_coated_metal, bare_polished_metal,
glossy_plastic, matte_plastic, uncertain. A paint/coating can hide the substrate;
use uncertain if its material cannot be judged from the image. Do not infer
physical roughness or metallic values. Return JSON only, exactly
{{"label":"one_allowed_label","evidence":"brief visual cue"}}.
Do not include color words in the evidence."""


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


def parse_response(response, allowed):
    content = response.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"VLM did not return JSON: {response}")
    result = json.loads(content[start:end + 1])
    if set(result) != {"label", "evidence"} or result["label"] not in allowed:
        raise ValueError(f"Invalid VLM label: {response}")
    if not isinstance(result["evidence"], str):
        raise ValueError(f"Invalid VLM evidence: {response}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.project_root = args.project_root.resolve()
    args.model_path = args.model_path.resolve()
    args.output_root = args.output_root.resolve()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    config = json.loads(args.config.read_text())
    labels = set(config["labels_to_renderer_presets"]) | {config["unmapped_label"]}
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
        mask_path = args.project_root / sample["mask"]
        crop_path = folder / "masked_crop.png"
        masked_crop(image_path, mask_path).save(crop_path)
        prompt = PROMPT.format(target=sample["target"])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(crop_path)},
            {"type": "text", "text": prompt},
        ]}]
        formatted = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[formatted], images=images, videos=videos,
                           padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, do_sample=False,
                                    max_new_tokens=config["max_new_tokens"])
        raw = processor.batch_decode(
            [ids[len(input_ids):] for input_ids, ids in zip(inputs.input_ids, output)],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        (folder / "raw_response.txt").write_text(raw + "\n")
        prediction = parse_response(raw, labels)
        label = prediction["label"]
        preset = config["labels_to_renderer_presets"].get(label)
        record = {
            "sample": sample, "source_image_sha256": sha(image_path),
            "mask_sha256": sha(mask_path), "masked_crop_sha256": sha(crop_path),
            "prompt": prompt, "raw_response": raw, "prediction": prediction,
            "renderer_preset": preset,
            "renderer_preset_origin": "predeclared semantic lookup, not image-derived PBR",
        }
        (folder / "prediction.json").write_text(json.dumps(record, indent=2) + "\n")
        if preset is not None:
            (folder / "renderer_preset.json").write_text(json.dumps({
                **preset, "semantic_label": label,
                "parameter_origin": "predeclared semantic lookup, not measured",
                "config_sha256": sha(args.config),
            }, indent=2) + "\n")
        results.append({"id": sample["id"], "label": label, "preset": preset})

    manifest = {
        "status": "complete", "model_id": config["model_id"],
        "model_snapshot": args.model_path.name,
        "model_config_sha256": sha(args.model_path / "config.json"),
        "config_sha256": sha(args.config), "script_sha256": sha(__file__),
        "project_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.project_root, text=True).strip(),
        "seed": config["seed"], "dtype": "float16", "do_sample": False,
        "max_new_tokens": config["max_new_tokens"], "results": results,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
