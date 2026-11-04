"""Discover object categories before grounding and segmenting individual instances."""

import json
from pathlib import Path
import re

from PIL import Image, ImageOps

from .vlm_initialization import MODEL_ID, REVISION, QwenPredictor, model_receipt, sha256, write_json


SCHEMA = "colorpeel_object_categories/v2"
POLICY = "whole_object_categories/v1"
KINDS = {"object", "background", "part", "text"}
PROMPT = """Inspect this image and classify the visible entities before selecting object categories.
Return exactly one JSON object with this schema:
{"entities": [{"category": "short category noun phrase", "kind": "object|background|part|text", "evidence": "brief visible evidence for this classification"}]}
For every distinct candidate category, assign exactly one kind and give at most
twelve words of visible evidence explaining that kind:
- object: an independently identifiable whole physical object with its own
  visible extent, including an identifiable partly occluded object.
- background: a scene region or enclosing surface such as wall, floor, ground,
  ceiling or sky. Surface boundaries, bricks or patterns do not make it an object.
- part: a component of another visible object, not another whole object.
- text: printed or attached letters, numbers or symbols as markings, not a
  separate physical object. Do not transcribe characters into category names.
Distinguish physical signs or number plates from the markings printed on them:
the whole plate can be an object if its own physical boundary is visible, while
the printed numbers themselves are text. A complete door can be an object;
being attached to a building does not automatically make a whole fixture a part.
Include both object and excluded background/part/text candidates so that the
selection is auditable. Include all visible whole-object categories, not just
the most prominent foreground object. List a category once even if several
instances are visible; use one consistent name, not synonyms for the same kind.
Use short English category nouns or noun phrases, with letters and spaces only.
Do not include colors, materials, textures, counts, positions or bounding boxes
in category names. Do not infer an object's identity from text printed on it.
Do not invent entities. If nothing is identifiable, return {"entities": []}.
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
    if not isinstance(response, dict) or set(response) != {"entities"}:
        raise ValueError("Require exactly one entities field")
    if not isinstance(response["entities"], list):
        raise ValueError("Entities must be a list")
    for entity in response["entities"]:
        if not isinstance(entity, dict) or set(entity) != {"category", "kind", "evidence"}:
            raise ValueError("Each entity requires exactly category, kind and evidence")
        if (not isinstance(entity["category"], str)
                or re.fullmatch(r"[A-Za-z]+(?: [A-Za-z]+)*", entity["category"]) is None):
            raise ValueError("Categories must be ordinary English noun phrases using letters and spaces")
        if not isinstance(entity["kind"], str) or entity["kind"] not in KINDS:
            raise ValueError("Entity kind must be object, background, part or text")
        if not isinstance(entity["evidence"], str) or not entity["evidence"].strip():
            raise ValueError("Entity evidence must be nonempty visible evidence")
    return response


def select_categories(response):
    categories, decisions = [], []
    for index, entity in enumerate(response["entities"]):
        category = entity["category"].lower()
        included = entity["kind"] == "object" and category not in categories
        reason = ("whole_object" if included else "duplicate_object_category"
                  if entity["kind"] == "object" else "excluded_" + entity["kind"])
        decisions.append({"entity_index": index, **entity, "included": included, "reason": reason})
        if included:
            categories.append(category)
    return {"policy": POLICY, "categories": categories, "decisions": decisions,
            "classification_quality_verified": False}


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
        selection = select_categories(response)
        write_json(output / "selection.json", selection)
        if not selection["categories"]:
            raise ValueError("No identifiable object categories; no segmentation can start")
        locks = {str(image_path): status["source"]["image_sha256"],
                 str(canonical): status["source"]["canonical_image_sha256"], **receipt["metadata_files"]}
        for name in ("prompt.txt", "raw_response.txt", "response.json", "selection.json", "model_receipt.json", "generation_receipt.json"):
            locks[str(output / name)] = sha256(output / name)
        if any(sha256(path) != digest for path, digest in locks.items()):
            raise ValueError("Input or provenance changed during object discovery")
        manifest = {"schema": SCHEMA, "status": "verified", "source": status["source"],
                    "categories": selection["categories"], "selection": selection,
                    "response": response, "source_locks": locks,
                    "vlm": {"model_id": MODEL_ID, "revision": REVISION, **predictor.runtime,
                            "model_receipt": str(output / "model_receipt.json")},
                    "scope": "Whole-object category selection only; grounding determines instances. Classification correctness and completeness are not verified."}
        path = output / "categories.json"
        write_json(path, manifest)
        status.update(status="verified", manifest_path=str(path))
        return path
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "status.json", status)
