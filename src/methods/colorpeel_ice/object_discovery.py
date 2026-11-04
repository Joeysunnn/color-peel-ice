"""Discover object categories before grounding and segmenting individual instances."""

import json
from pathlib import Path
import re

from PIL import Image, ImageOps

from .vlm_initialization import MODEL_ID, REVISION, QwenPredictor, model_receipt, sha256, write_json


SCHEMA = "colorpeel_object_categories/v1"
PROMPT = """Inspect this image and list every visible, independently identifiable object category.
Return exactly one JSON object with this schema:
{"categories": ["object category", "another object category"]}
Use short English category nouns or noun phrases, with letters and spaces only.
Name whole objects, including identifiable partly occluded objects; do not list
their parts separately. Include all visible object categories, not just the most
prominent foreground object. List a category only once even if several instances
are visible. Use one consistent category name rather than synonyms for the same
kind of object. Do not include colors, materials, textures, counts, positions or
bounding boxes in category names. Exclude walls, floors, ground, sky and other
background regions. Do not infer an object's identity from printed text.
If no object is identifiable, return {"categories": []}. Do not invent objects.
Return JSON without Markdown or surrounding commentary."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_response(raw):
    text = raw.strip()
    if text.startswith("```"):
        fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if fence is None:
            raise ValueError("Require one complete JSON code fence")
        text = fence.group(1)
    response = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(response, dict) or set(response) != {"categories"}:
        raise ValueError("Require exactly one categories field")
    categories = response["categories"]
    if not isinstance(categories, list):
        raise ValueError("Categories must be a list")
    if any(not isinstance(category, str)
           or re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+)*", category) is None
           for category in categories):
        raise ValueError("Categories must be ordinary English noun phrases using letters and spaces")
    if len({category.lower() for category in categories}) != len(categories):
        raise ValueError("Object categories must be unique")
    return response


def run(image_path, model_dir, output, device="cuda:0"):
    image_path, model_dir, output = [Path(path).resolve() for path in (image_path, model_dir, output)]
    repo = Path(__file__).resolve().parents[3]
    if output.exists() or any(output.is_relative_to(root) for root in (repo, model_dir)):
        raise ValueError("Require a fresh external output directory outside source/model directories")
    output.mkdir(parents=True)
    status = {"schema": SCHEMA, "status": "started", "source": {"image": str(image_path)}}
    try:
        status["source"]["image_sha256"] = sha256(image_path)
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
            status["source"].update(original_size=list(opened.size), input_size=list(image.size),
                                      conversion="EXIF transpose, RGB; no crop or mask")
        canonical = output / "source.png"
        image.save(canonical)
        status["source"].update(canonical_image=str(canonical), canonical_image_sha256=sha256(canonical))
        (output / "prompt.txt").write_text(PROMPT, encoding="utf8")
        receipt = model_receipt(model_dir)
        write_json(output / "model_receipt.json", receipt)
        predictor = QwenPredictor(model_dir, device, prompt=PROMPT)
        raw = predictor(image)
        (output / "raw_response.txt").write_text(raw, encoding="utf8")
        write_json(output / "generation_receipt.json", predictor.runtime)
        if predictor.runtime.get("reached_token_limit"):
            raise ValueError("Qwen response reached its token limit; raw output preserved")
        response = parse_response(raw)
        write_json(output / "response.json", response)
        if not response["categories"]:
            raise ValueError("No identifiable object categories; no segmentation can start")
        locks = {str(image_path): status["source"]["image_sha256"],
                 str(canonical): status["source"]["canonical_image_sha256"], **receipt["metadata_files"]}
        for name in ("prompt.txt", "raw_response.txt", "response.json", "model_receipt.json", "generation_receipt.json"):
            locks[str(output / name)] = sha256(output / name)
        if any(sha256(path) != digest for path, digest in locks.items()):
            raise ValueError("Input or provenance changed during object discovery")
        manifest = {"schema": SCHEMA, "status": "verified", "source": status["source"],
                    "categories": [category.lower() for category in response["categories"]],
                    "response": response, "source_locks": locks,
                    "vlm": {"model_id": MODEL_ID, "revision": REVISION, **predictor.runtime,
                            "model_receipt": str(output / "model_receipt.json")},
                    "scope": "Category discovery only; grounding determines instances. Completeness is not verified."}
        path = output / "categories.json"
        write_json(path, manifest)
        status.update(status="verified", manifest_path=str(path))
        return path
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "status.json", status)
