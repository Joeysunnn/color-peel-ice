"""Validation helpers for textual-inversion initializer tokens."""

import copy
import hashlib
import json
from pathlib import Path
import re


BRANCH_TOKENS = {"subject": "<S*>", "color": "<C*>", "material": "<M*>", "texture": "<T*>"}
MATERIAL_ABLATION_SCHEMA = "colorpeel_material_initializer_ablation/v1"
MATERIAL_ABLATION_INITIALIZERS = ("painted", "glossy", "material")


def single_token_initializer_id(tokenizer, initializer_token: str) -> int:
    """Return the token id, rejecting initializers split into multiple BPE pieces."""

    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Initializer token {initializer_token!r} must encode to exactly one token; "
            f"got {len(token_ids)} token ids: {token_ids}"
        )
    return int(token_ids[0])


def initializer_candidates(branch, semantic):
    """Prefer Qwen's single-word Subject category without rewriting its raw candidates."""
    if branch not in BRANCH_TOKENS:
        raise ValueError("require a supported initializer branch")
    candidates = list(semantic["candidates"])
    category = semantic.get("category")
    if (branch == "subject" and isinstance(category, str) and re.fullmatch(r"[A-Za-z]+", category)
            and (not candidates or candidates[0] != category)):
        return list(dict.fromkeys([category, *candidates]))
    return candidates


def select_initializer(tokenizer, candidates):
    """Audit every candidate; select the first ordinary, single-word, single-BPE word."""
    if not isinstance(candidates, list):
        raise ValueError("initializer candidates must be a list")
    vocabulary = tokenizer.get_vocab()
    modifier_ids = {vocabulary[token] for token in BRANCH_TOKENS.values() if token in vocabulary}
    special_ids = set(tokenizer.all_special_ids)
    selected_word, selected_id, audit = None, None, []
    for candidate in candidates:
        word = candidate.strip() if isinstance(candidate, str) else None
        reasons, token_ids = [], []
        if word is None:
            reasons.append("invalid_type")
        elif not word:
            reasons.append("empty")
        elif word.lower() in ("unknown", "none", "null", "unclear"):
            reasons.append("unknown_value")
        elif re.fullmatch(r"[A-Za-z]+", word) is None:
            reasons.append("not_single_word")
        else:
            token_ids = [int(value) for value in tokenizer.encode(word, add_special_tokens=False)]
            if len(token_ids) != 1:
                reasons.append("multiple_bpe")
            if tokenizer.unk_token_id in token_ids:
                reasons.append("unknown_token")
            if special_ids.intersection(token_ids):
                reasons.append("special_token")
            if modifier_ids.intersection(token_ids):
                reasons.append("modifier_token")
        accepted = not reasons
        audit.append({"candidate": candidate, "word": word, "token_ids": token_ids,
                      "accepted": accepted, "reasons": reasons})
        if accepted and selected_word is None:
            selected_word, selected_id = word, token_ids[0]
    return selected_word, selected_id, audit


def _checked_locks(files):
    if not isinstance(files, dict) or not files:
        raise ValueError("initializer manifest requires nonempty file locks")
    checked = {}
    for name, digest in files.items():
        path = Path(name)
        if (not path.is_absolute() or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or path.suffix.lower() in (".bin", ".safetensors", ".ckpt", ".pt", ".pth")):
            raise ValueError("initializer locks must be absolute paths with SHA256, excluding model weights")
        resolved = path.resolve()
        if resolved in checked or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"initializer source lock is duplicate or changed: {path}")
        checked[resolved] = digest
    return checked


def load_initializer_manifest(path, tokenizer, model_root):
    """Recheck the frozen four-branch VLM decision against actual SD2.1 tokenization."""
    manifest = json.loads(Path(path).read_bytes())
    if isinstance(manifest, dict) and manifest.get("schema") == MATERIAL_ABLATION_SCHEMA:
        if not isinstance(manifest.get("ablation"), dict) or not isinstance(manifest.get("parent_manifest"), str):
            raise ValueError("controlled Material initializer requires a parent and explicit selection")
        expected = build_material_ablation_manifest(
            manifest["parent_manifest"], tokenizer, model_root, manifest["ablation"].get("initializer"),
        )
        if manifest != expected:
            raise ValueError("controlled Material initializer manifest differs from its verified parent or selection")
        return manifest
    if (not isinstance(manifest, dict) or manifest.get("schema") not in (
            "colorpeel_vlm_initializers/v1", "colorpeel_vlm_initializers/v2")
            or manifest.get("status") != "verified"):
        raise ValueError("require a verified colorpeel_vlm_initializers/v1 or v2 manifest")
    locks = _checked_locks(manifest.get("source_locks"))
    source, vlm, text = (manifest.get(key, {}) for key in ("source", "vlm", "tokenizer"))
    if not all(isinstance(value, dict) for value in (source, vlm, text)):
        raise ValueError("source, vlm and tokenizer must be manifest objects")
    image = Path(source.get("image", ""))
    if (not image.is_absolute() or image.resolve() not in locks
            or locks[image.resolve()] != source.get("image_sha256")):
        raise ValueError("source image and its SHA256 must occur in source_locks")
    if any(not isinstance(vlm.get(key), str) or not vlm[key].strip() for key in ("model_id", "revision")):
        raise ValueError("record the VLM model ID and revision")
    root = Path(model_root).resolve()
    recorded_root = Path(text.get("model_root", ""))
    if (not recorded_root.is_absolute() or recorded_root.resolve() != root
            or text.get("model_max_length") != 77 or tokenizer.model_max_length != 77):
        raise ValueError("initializer tokenizer root or CLIP77 length differs")
    tokenizer_locks = _checked_locks(text.get("files"))
    actual_files = {item.resolve() for item in (root / "tokenizer").rglob("*") if item.is_file()}
    if (not actual_files or set(tokenizer_locks) != actual_files
            or any(locks.get(item) != digest for item, digest in tokenizer_locks.items())):
        raise ValueError("all actual tokenizer files must be locked, with matching source_locks")
    branches = manifest.get("branches")
    if not isinstance(branches, dict) or set(branches) != set(BRANCH_TOKENS):
        raise ValueError("initializer manifest must contain exactly four named branches")
    folder = Path(path).resolve().parent
    response_paths = [folder / name for name in ("raw_response.txt", "response.json")]
    if any(item.resolve() not in locks for item in response_paths):
        raise ValueError("raw_response.txt and response.json must be locked beside the manifest")
    raw, response = (json.loads(item.read_bytes()) for item in response_paths)
    if (not isinstance(raw, dict) or set(raw) != set(BRANCH_TOKENS)
            or raw != response or raw != manifest.get("response")):
        raise ValueError("raw VLM response, parsed response and manifest response must agree")
    if manifest['schema'] == 'colorpeel_vlm_initializers/v2':
        from src.methods.colorpeel_ice.vlm_initialization import build_branches, parse_response
        parsed = parse_response(response_paths[0].read_text(encoding='utf8'))
        expected, failures = build_branches(parsed, tokenizer)
        if failures or branches != expected:
            raise ValueError('controlled initializer policy, semantic evidence or token audit differs')
        if manifest.get('subject_category') != parsed['subject']['category']:
            raise ValueError('subject_category differs from the locked VLM response')
        return manifest
    for branch, token in BRANCH_TOKENS.items():
        entry = branches[branch]
        if (not isinstance(entry, dict) or entry.get("token") != token
                or not isinstance(entry.get("description"), str) or not entry["description"].strip()
                or entry.get("confidence") not in ("high", "medium", "low")
                or not isinstance(entry.get("candidates"), list) or not entry["candidates"]
                or not all(isinstance(value, str) for value in entry["candidates"])):
            raise ValueError(f"invalid or unknown initializer decision for {branch}")
        semantic_fields = {"description", "confidence", "candidates"}
        if branch == "subject":
            semantic_fields.add("category")
        elif branch == "material":
            semantic_fields.update(("substrate", "visible_finish"))
        if (not isinstance(raw[branch], dict) or not semantic_fields.issubset(raw[branch])
                or any(field not in entry or entry[field] != raw[branch][field] for field in semantic_fields)):
            raise ValueError(f"{branch} semantic fields differ from the locked VLM response")
        candidates = initializer_candidates(branch, entry)
        if entry.get("selection_candidates") != candidates:
            raise ValueError(f"selection candidates differ from the category policy for {branch}")
        word, token_id, audit = select_initializer(tokenizer, candidates)
        if (word is None or entry.get("initializer") != word
                or type(entry.get("initializer_token_id")) is not int or entry["initializer_token_id"] != token_id
                or entry.get("candidate_audit") != audit):
            raise ValueError(f"initializer choice or candidate audit differs for {branch}")
    if manifest.get("subject_category") != raw["subject"]["category"]:
        raise ValueError("subject_category differs from the locked VLM response")
    return manifest


def build_material_ablation_manifest(parent_path, tokenizer, model_root, initializer):
    """Record an explicit Material-only intervention without changing Qwen evidence."""
    if initializer not in MATERIAL_ABLATION_INITIALIZERS:
        raise ValueError("controlled Material initializer must be painted, glossy or material")
    parent_path = Path(parent_path)
    if not parent_path.is_absolute():
        raise ValueError("controlled Material initializer requires an absolute parent manifest")
    parent_path = parent_path.resolve()
    parent_bytes = parent_path.read_bytes()
    parent_record = json.loads(parent_bytes)
    if (not isinstance(parent_record, dict)
            or parent_record.get("schema") != "colorpeel_vlm_initializers/v1"):
        raise ValueError("controlled Material initializer requires an original VLM-v1 parent")
    parent = load_initializer_manifest(parent_path, tokenizer, model_root)
    word, token_id, audit = select_initializer(tokenizer, [initializer])
    if word != initializer:
        raise ValueError("controlled Material initializer must be one ordinary tokenizer token")
    parent_sha = hashlib.sha256(parent_bytes).hexdigest()
    manifest = copy.deepcopy(parent)
    manifest.update(schema=MATERIAL_ABLATION_SCHEMA, selection_origin="controlled_ablation",
                    parent_manifest=str(parent_path), parent_manifest_sha256=parent_sha,
                    ablation={"branch": "material", "initializer": initializer})
    manifest["source_locks"][str(parent_path)] = parent_sha
    manifest["branches"]["material"].update(
        initializer=word, initializer_token_id=token_id,
        selection_origin="controlled_ablation", selection_candidates=[initializer], candidate_audit=audit,
    )
    return manifest


def validate_initializer_training(args, manifest):
    """Opt-in scope; callers with no manifest retain the historical trainer guards."""
    token, initializer = args.modifier_token, args.initializer_token
    branch = next((name for name, value in BRANCH_TOKENS.items() if value == token), None)
    if manifest.get("schema") == MATERIAL_ABLATION_SCHEMA and branch != "material":
        raise ValueError("controlled initializer ablation is restricted to the Material branch")
    if (branch is None or initializer != manifest["branches"][branch]["initializer"]
            or args.subject_lora_mode != "token_local_kv" or args.subject_lora_rank != 64
            or args.subject_lora_alpha != 64 or args.freeze_model != "crossattn_kv"
            or args.token_local_kv or args.perfusion_subject or args.enable_xformers_memory_efficient_attention
            or args.k_learning_rate is not None or args.v_learning_rate is not None
            or args.adam_weight_decay != 0 or args.use_8bit_adam or args.resume_from_checkpoint is not None
            or args.with_prior_preservation or args.joint_two_object_binding or args.strict_unpaired_modifier_updates
            or args.cos_weight != 0 or args.lambda_attention != 0
            or args.texture_background_preservation_weight != 0 or args.texture_external_preservation_weight != 0
            or args.concept_isolation_weight != 0):
        raise ValueError("VLM initialization requires one matching token, fresh token-local LoRA64/alpha64, decay0, no other objectives")
    return branch


def initializer_vector_audit(base, initializer_after, modifier_before, modifier_after):
    """Verify and fingerprint the actual copy, without changing tensors or RNG."""
    import torch

    vectors = {"base_initializer": base, "initializer_after_copy": initializer_after,
               "modifier_before_copy": modifier_before, "modifier_after_copy": modifier_after}
    if any(tuple(value.shape) != (1024,) or value.dtype != torch.float32
           or not bool(torch.isfinite(value).all()) for value in vectors.values()):
        raise ValueError("VLM initializer copy requires finite FP32 SD2.1 vectors of width1024")
    if not torch.equal(base, initializer_after) or not torch.equal(base, modifier_after):
        raise ValueError("modifier initialization did not exactly copy the unchanged base initializer")
    return {"shape": [1024], "dtype": "float32", "initialized_equal_base": True,
            "initializer_unchanged": True,
            **{name + "_sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
               for name, value in vectors.items()}}
