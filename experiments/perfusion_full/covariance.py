"""Estimate the uncentered CLIP text covariance used by full Perfusion.

The paper specifies 100,000 LAION captions but does not specify how to reduce a
caption's contextual token sequence to one vector. This project takes one
content-token vector per caption: a seeded hash chooses uniformly among the
non-special positions retained by the SD 1.4 tokenizer. Sampling captions also
uses seeded hash priorities, so input order and duplicate rows cannot change
the selected set. The input file must contain LAION captions; this tool does
not download or substitute a caption source.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path

import torch


WEIGHT_NAME = "covariance.pt"
INVERSE_NAME = "covariance_inverse.pt"
PROVENANCE_NAME = "provenance.json"


def _digest(domain: str, seed: int, caption: str) -> bytes:
    payload = domain.encode("ascii") + b"\0" + str(seed).encode("ascii") + b"\0" + caption.encode("utf-8")
    return hashlib.sha256(payload).digest()


def _caption(line: bytes, jsonl: bool) -> str:
    decoded = line.decode("utf-8-sig").strip()
    if not decoded:
        return ""
    if not jsonl:
        return decoded
    row = json.loads(decoded)
    if not isinstance(row, dict):
        raise ValueError("each JSONL row must be an object")
    for key in ("caption", "text", "TEXT"):
        if key in row:
            if not isinstance(row[key], str):
                raise ValueError(f"JSONL {key} must be a string")
            return row[key].strip()
    raise ValueError("each JSONL row must contain caption, text, or TEXT")


def sample_distinct_captions(path: Path, count: int, seed: int) -> tuple[list[str], dict]:
    """Keep the smallest seeded hash priorities among distinct nonempty captions."""
    if count <= 0:
        raise ValueError("count must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    path = Path(path)
    if path.suffix.lower() not in {".jsonl", ".txt", ".text"}:
        raise ValueError("caption source must be .jsonl, .txt, or .text")
    source_hash = hashlib.sha256()
    heap: list[tuple[int, str]] = []
    retained: set[str] = set()
    rows = nonempty_rows = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            source_hash.update(raw_line)
            rows += 1
            caption = _caption(raw_line, path.suffix.lower() == ".jsonl")
            if not caption:
                continue
            nonempty_rows += 1
            if caption in retained:
                continue
            priority = int.from_bytes(_digest("sample", seed, caption), "big")
            if len(heap) == count:
                if priority >= -heap[0][0]:
                    continue
                _, removed = heapq.heapreplace(heap, (-priority, caption))
                retained.remove(removed)
            else:
                heapq.heappush(heap, (-priority, caption))
            retained.add(caption)
    if len(heap) != count:
        raise ValueError(f"source has fewer than {count} distinct nonempty captions: {len(heap)}")
    selected = [caption for _, caption in sorted(heap, key=lambda item: -item[0])]
    sample_hash = hashlib.sha256()
    for caption in selected:
        sample_hash.update(caption.encode("utf-8") + b"\n")
    return selected, {
        "source_sha256": source_hash.hexdigest(),
        "source_rows": rows,
        "source_nonempty_rows": nonempty_rows,
        "selected_distinct_count": len(selected),
        "selected_captions_sha256": sample_hash.hexdigest(),
        "caption_sampling": "smallest SHA256(sample\\0seed\\0stripped_caption), unique captions",
    }


def content_token_positions(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                            special_ids: set[int], captions: list[str], seed: int) -> torch.Tensor:
    """Return one seeded content-token position per caption in a tokenized batch."""
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or input_ids.shape[0] != len(captions):
        raise ValueError("tokenized batch has unexpected shape")
    positions = []
    for row, caption in enumerate(captions):
        valid = [column for column, token_id in enumerate(input_ids[row].tolist())
                 if bool(attention_mask[row, column]) and int(token_id) not in special_ids]
        if not valid:
            raise ValueError(f"caption has no content token after tokenization: {caption[:80]!r}")
        choice = int.from_bytes(_digest("token", seed, caption), "big") % len(valid)
        positions.append(valid[choice])
    return torch.tensor(positions, dtype=torch.long, device=input_ids.device)


def inverse_with_minimal_ridge(covariance: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Invert an uncentered covariance, adding only the ridge needed for 1e8 conditioning."""
    covariance = covariance.to(dtype=torch.float64, device="cpu")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    covariance = (covariance + covariance.T) / 2
    eigenvalues = torch.linalg.eigvalsh(covariance)
    smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    if largest <= 0 or not torch.isfinite(eigenvalues).all():
        raise ValueError("covariance must have finite positive energy")
    floor = largest * 1e-8
    ridge = max(0.0, floor - smallest)
    adjusted = covariance + ridge * torch.eye(covariance.shape[0], dtype=torch.float64)
    inverse = torch.linalg.inv(adjusted)
    inverse = (inverse + inverse.T) / 2
    residual = torch.linalg.matrix_norm(adjusted @ inverse - torch.eye(covariance.shape[0]), ord=float("inf"))
    diagnostics = {
        "covariance_min_eigenvalue": smallest,
        "covariance_max_eigenvalue": largest,
        "ridge": ridge,
        "ridge_rule": "smallest nonnegative ridge yielding min eigenvalue >= max eigenvalue * 1e-8",
        "regularized_condition_number": (largest + ridge) / (smallest + ridge),
        "inverse_identity_residual_inf": float(residual),
        "covariance_trace": float(torch.trace(covariance)),
    }
    return inverse, diagnostics


def estimate_covariance(captions: list[str], model: str, seed: int, device: str,
                        batch_size: int = 32) -> tuple[torch.Tensor, dict]:
    """Average x x^T for one frozen contextual CLIP content token per caption."""
    if not captions or batch_size <= 0:
        raise ValueError("captions and batch_size must be nonempty")
    from transformers import AutoTokenizer, CLIPTextModel

    tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", use_fast=False)
    encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder").to(device)
    encoder.eval().requires_grad_(False)
    width = int(encoder.config.hidden_size)
    if width != 768:
        raise ValueError(f"expected SD 1.4 CLIP width 768, got {width}")
    if tokenizer.model_max_length != 77:
        raise ValueError(f"expected SD 1.4 CLIP context 77, got {tokenizer.model_max_length}")
    special_ids = set(int(value) for value in tokenizer.all_special_ids)
    accumulator = torch.zeros((width, width), dtype=torch.float64)
    with torch.inference_mode():
        for start in range(0, len(captions), batch_size):
            batch_captions = captions[start:start + batch_size]
            encoded = tokenizer(batch_captions, padding="max_length", truncation=True,
                                max_length=77, return_tensors="pt")
            ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            positions = content_token_positions(ids, mask, special_ids, batch_captions, seed)
            hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            vectors = hidden[torch.arange(len(batch_captions), device=ids.device), positions].to(
                dtype=torch.float64, device="cpu")
            accumulator += vectors.T @ vectors
    covariance = accumulator / len(captions)
    return covariance, {
        "model": model,
        "model_commit_hash": getattr(encoder.config, "_commit_hash", None),
        "text_encoder_config_sha256": hashlib.sha256(encoder.config.to_json_string().encode()).hexdigest(),
        "tokenizer_vocab_sha256": hashlib.sha256(
            json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "text_width": width,
        "context_length": 77,
        "estimator": "uncentered_mean_outer_product",
        "token_selection": "seeded SHA256(token\\0seed\\0caption) modulo non-special retained token count",
        "encoder_mode": "eval, frozen, float32; covariance accumulated in float64 on CPU",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.count != 100000:
        raise ValueError("full Perfusion covariance requires exactly 100000 distinct captions")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    captions, source = sample_distinct_captions(args.captions, args.count, args.seed)
    covariance, encoder = estimate_covariance(captions, args.model, args.seed, args.device, args.batch_size)
    inverse, diagnostics = inverse_with_minimal_ridge(covariance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(covariance, args.output_dir / WEIGHT_NAME)
    torch.save(inverse, args.output_dir / INVERSE_NAME)
    provenance = {
        "schema": "perfusion_full_covariance/v1",
        "captions_path": str(args.captions.resolve()),
        "seed": args.seed,
        "requested_count": args.count,
        "batch_size": args.batch_size,
        "device": args.device,
        **source, **encoder, **diagnostics,
        "covariance_sha256": hashlib.sha256((args.output_dir / WEIGHT_NAME).read_bytes()).hexdigest(),
        "inverse_sha256": hashlib.sha256((args.output_dir / INVERSE_NAME).read_bytes()).hexdigest(),
    }
    (args.output_dir / PROVENANCE_NAME).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved covariance of {len(captions)} distinct captions to {args.output_dir}")


if __name__ == "__main__":
    main()
