import json
import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[3] / "src" / "train" / "context_prior_utils.py"
SPEC = importlib.util.spec_from_file_location("context_prior_utils", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
load_context_prior_records = MODULE.load_context_prior_records


def test_load_context_prior_records_resolves_images_and_rejects_modifier_prompts(tmp_path):
    image = tmp_path / "mailbox.png"
    image.write_bytes(b"placeholder")
    manifest = tmp_path / "class_prior_manifest.jsonl"
    manifest.write_text(
        json.dumps({"image_path": "mailbox.png", "prompt": "a photo of a mailbox in snow"}) + "\n",
        encoding="utf-8",
    )
    assert load_context_prior_records(manifest, forbidden_tokens=("<S*>",)) == [
        (image, "a photo of a mailbox in snow")
    ]
    manifest.write_text(
        json.dumps({"image_path": "mailbox.png", "prompt": "a photo of <S*> mailbox in snow"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="modifier token"):
        load_context_prior_records(manifest, forbidden_tokens=("<S*>",))
