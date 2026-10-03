"""Blind qualitative appearance audit of generated material transfer images."""

import argparse
import hashlib
import json
from pathlib import Path


PROMPT = """Look at the main object only. Do not infer its material from the object
name, and do not use text outside the image. Return JSON with exactly these keys:
{"metallic_appearance":"clear_metallic|weak_or_uncertain|nonmetal",
"gloss":"matte|semi_gloss|glossy|uncertain",
"object_kind":"cone|mailbox|other|uncertain",
"dominant_color":"gray|red|blue|other|uncertain",
"evidence":"one brief visual cue about reflections"}.
Judge visible reflections, not hidden substrate."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(raw):
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON: {raw}")
    data = json.loads(raw[start:end + 1])
    allowed = {
        "metallic_appearance": {"clear_metallic", "weak_or_uncertain", "nonmetal"},
        "gloss": {"matte", "semi_gloss", "glossy", "uncertain"},
        "object_kind": {"cone", "mailbox", "other", "uncertain"},
        "dominant_color": {"gray", "red", "blue", "other", "uncertain"},
    }
    if set(data) != set(allowed) | {"evidence"}:
        raise ValueError(f"Wrong fields: {raw}")
    for key, values in allowed.items():
        if data[key] not in values:
            raise ValueError(f"Wrong {key}: {raw}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows = [json.loads(line) for line in (args.evaluation / "manifest.jsonl").read_text().splitlines()]
    from PIL import Image
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    with (args.output / "predictions.jsonl").open("w") as ledger:
        for row in rows:
            path = args.evaluation / "images" / f'{row["id"]}.png'
            with Image.open(path) as image:
                image.verify()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(path)},
                {"type": "text", "text": PROMPT},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos,
                               padding=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=128)
            raw = processor.batch_decode(
                [ids[len(input_ids):] for input_ids, ids in zip(inputs.input_ids, output)],
                skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            try:
                label, error = parse(raw), None
            except (ValueError, json.JSONDecodeError) as exc:
                label, error = None, str(exc)
            ledger.write(json.dumps({**row, "image_sha256": sha(path),
                                     "prediction": label, "raw_response": raw,
                                     "parse_error": error}) + "\n")
            ledger.flush()
    (args.output / "provenance.json").write_text(json.dumps({
        "status": "complete", "method": "Qwen3-VL appearance annotation, not physical R/M ground truth",
        "evaluation_manifest_sha256": sha(args.evaluation / "manifest.jsonl"),
        "model_snapshot": args.model_path.name,
        "model_config_sha256": sha(args.model_path / "config.json"),
        "prompt": PROMPT, "script_sha256": sha(__file__), "seed": 42,
        "dtype": "float16", "do_sample": False, "rows": len(rows),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
