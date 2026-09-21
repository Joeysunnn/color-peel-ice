"""Validation helpers for prompt-conditional class-prior image manifests."""

from __future__ import annotations

import json
from pathlib import Path


def load_context_prior_records(manifest_path, forbidden_tokens=()):
    """Return validated ``(image_path, prompt)`` records from a JSONL manifest."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"class prior manifest does not exist: {manifest_path}")
    records = []
    seen_image_paths = set()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid class prior JSON at {manifest_path}:{line_number}") from error
        if not isinstance(value, dict) or not isinstance(value.get("image_path"), str) or not isinstance(value.get("prompt"), str):
            raise ValueError(f"class prior record requires string image_path and prompt: {manifest_path}:{line_number}")
        image_path = Path(value["image_path"])
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"class prior image does not exist: {image_path}")
        image_path = image_path.resolve()
        if image_path in seen_image_paths:
            raise ValueError(f"class prior image is duplicated: {image_path}")
        seen_image_paths.add(image_path)
        prompt = value["prompt"]
        if any(token in prompt for token in forbidden_tokens):
            raise ValueError(f"class prior prompt contains a modifier token: {manifest_path}:{line_number}")
        records.append((image_path, prompt))
    if not records:
        raise ValueError(f"class prior manifest has no records: {manifest_path}")
    return records
