"""Replay the Phase 1 Texture/Material factorial evaluation on frozen adapters."""

import argparse
import csv
import gc
import hashlib
import html
import itertools
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd15_token_local_kv_v1.evaluate import PROTOCOL as SD15_PROTOCOL, load_pipeline
from experiments.sd_native_comparison_v1.generate import package_versions, write_json
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

PROTOCOL = Path(__file__).with_name("protocol.json")
METHODS = ("fullrank", "lora_r64")
TOKENS = {"texture": "<T*>", "material": "<M*>"}
WEIGHTS = {"fullrank": "pytorch_token_local_kv_weights.bin",
           "lora_r64": "pytorch_lora_kv_weights.bin"}
AXES = ("texture", "material", "shape", "color", "light", "background")


def protocol_hash(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def target_ids(protocol, branch):
    return [key for key, value in protocol[branch + "s"].items() if value["role"] == "target"]


def build_rows(protocol):
    rows = []
    for method, texture, material in itertools.product(
            METHODS, target_ids(protocol, "texture"), target_ids(protocol, "material")):
        for context, condition, seed in itertools.product(
                protocol["dataset"]["heldout_contexts"], protocol["inference"]["conditions"],
                protocol["inference"]["seeds"]):
            t = "<T*>" if condition in {"texture_only", "both"} else protocol["textures"][texture]["literal"]
            m = "<M*>" if condition in {"material_only", "both"} else protocol["materials"][material]["literal"]
            prompt = (f"a photo of a {context['color']} {context['shape']} with {t} texture "
                      f"made of {m}, {protocol['dataset']['light_phrases'][context['light']]}")
            identity = f"{method}__{texture}__{material}__{context['id']}__{condition}__seed{seed}"
            rows.append({"id": identity, "method": method, "texture": texture, "material": material,
                         "context": context["id"], "shape": context["shape"], "color": context["color"],
                         "light": context["light"], "condition": condition, "seed": seed,
                         "prompt": prompt, "image_path": f"images/{identity}.png"})
    if len(rows) != 256 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Phase 1 requires the locked 256-image evaluation matrix")
    return rows


def verify_inputs(index, protocol):
    if index.get("mode") != "pilot":
        raise ValueError("Phase 1 inference requires completed pilot training, not smoke training")
    if protocol["base_model"] != read_json(SD15_PROTOCOL)["model"]["root"]:
        raise ValueError("Phase 1 base differs from the shared SD1.5 pipeline loader")
    if index.get("protocol_sha256") != protocol_hash(protocol):
        raise ValueError("training index protocol differs")
    expected = {(method, branch, concept) for method in METHODS for branch in TOKENS
                for concept in target_ids(protocol, branch)}
    runs = index.get("runs", [])
    keys = [(run.get("method"), run.get("branch"), run.get("concept_id")) for run in runs]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("training index must contain exactly eight distinct concept/method runs")
    for run in runs:
        branch, method = run["branch"], run["method"]
        if (run.get("status") != "verified" or run.get("token") != TOKENS[branch]
                or run.get("weight") != WEIGHTS[method]
                or run.get("steps") != protocol["training"]["steps"]):
            raise ValueError("training index token, method, pilot steps or verification differs")
        checkpoint = Path(run["checkpoint"])
        hashes = run.get("hashes", {})
        required = {str(checkpoint / name) for name in
                    (WEIGHTS[method], TOKENS[branch] + ".bin", "adaptation_config.json")}
        if (not checkpoint.is_absolute() or not Path(run["run"]).is_absolute()
                or not required <= set(hashes)):
            raise ValueError("training index lacks absolute checkpoint paths or adapter hashes")
        for path, digest in hashes.items():
            if not Path(path).is_absolute() or sha256(Path(path)) != digest:
                raise ValueError(f"training artifact changed: {path}")


def attention_kwargs(pipe, prompt, guidance):
    """T uses the existing primary slot; the labels do not change its semantics."""
    import torch
    ids = pipe.tokenizer(prompt, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to(pipe.unet.device)
    masks = {}
    for token, label in (("<T*>", "subject"), ("<M*>", "material")):
        token_id = pipe.tokenizer.convert_tokens_to_ids(token)
        if token_id == pipe.tokenizer.unk_token_id:
            raise ValueError(f"missing registered token: {token}")
        mask = ids == token_id
        if prompt.count(token) > 1 or int(mask.sum()) != prompt.count(token):
            raise ValueError("learned token was repeated or truncated")
        masks[label] = torch.cat([torch.zeros_like(mask), mask]) if guidance > 1 else mask
    return {"modifier_token_mask": masks}


def contact_sheets(output, rows, protocol):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default()
    by_id = {row["id"]: row for row in rows}
    sheets = []
    pairs = list(itertools.product(target_ids(protocol, "texture"), target_ids(protocol, "material")))
    conditions = protocol["inference"]["conditions"]
    for method, context, seed in itertools.product(
            METHODS, protocol["dataset"]["heldout_contexts"], protocol["inference"]["seeds"]):
        sheet = Image.new("RGB", (4 * 272, 56 + 4 * 302), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"{method} / {context['id']} / seed {seed}", fill="black", font=font)
        for col, condition in enumerate(conditions):
            draw.text((col * 272 + 8, 32), condition, fill="black", font=font)
        for line, (texture, material) in enumerate(pairs):
            for col, condition in enumerate(conditions):
                identity = f"{method}__{texture}__{material}__{context['id']}__{condition}__seed{seed}"
                row = by_id[identity]
                with Image.open(output / row["image_path"]) as source:
                    thumbnail = source.convert("RGB").resize((256, 256))
                x, y = col * 272 + 8, 56 + line * 302
                sheet.paste(thumbnail, (x, y))
                draw.text((x, y + 258), texture, fill="black", font=font)
                draw.text((x, y + 271), material, fill="black", font=font)
                if row["status"] == "safety_filtered":
                    draw.text((x, y + 284), "SAFETY FILTERED", fill="red", font=font)
        path = f"contact_sheets/{method}__{context['id']}__seed{seed}.jpg"
        (output / path).parent.mkdir(exist_ok=True)
        sheet.save(output / path, quality=90)
        sheets.append(path)
    return sheets


def review_bundle(output, rows, protocol):
    sheets = contact_sheets(output, rows, protocol)
    shuffled = list(rows)
    random.Random(protocol["seed"]).shuffle(shuffled)
    key, cards = [], []
    with (output / "blinded_ratings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_id", *AXES, "notes"])
        writer.writeheader()
        for index, row in enumerate(shuffled):
            blind_id = f"sample_{index:04d}"
            key.append({"blind_id": blind_id, "id": row["id"], "image_path": row["image_path"],
                        "status": row["status"]})
            if row["status"] != "safety_filtered":
                writer.writerow({"blind_id": blind_id})
                cards.append(f'<figure><img width="256" src="{html.escape(row["image_path"], quote=True)}">'
                             f'<figcaption>{blind_id}</figcaption></figure>')
    write_json(output / "blinded_key.json", {"samples": key})
    (output / "blinded_review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Blinded Phase 1 review</title>'
        '<style>body{display:flex;flex-wrap:wrap}figure{margin:8px}</style>'
        '<p>Describe the observed texture, material, shape, color, lighting and background in blinded_ratings.csv. '
        'Use blinded_key.json only after completing the descriptions. Filtered images are excluded.</p>' + "".join(cards), encoding="utf-8")
    links = "".join(f'<li><a href="{html.escape(path, quote=True)}">{html.escape(path)}</a></li>' for path in sheets)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Phase 1 T/M evaluation</title>'
        '<h1>Phase 1 Texture/Material pilot</h1><p>Columns: literal attributes, learned T with literal M, '
        'literal T with learned M, learned T and M. Identical context and seed within every row.</p>'
        '<p>Filtered images are marked. Contact sheets support visual comparison, not measured BRDF recovery.</p>'
        '<a href="blinded_review.html">Blinded review</a><ul>' + links + '</ul>', encoding="utf-8")
    return sheets


def run(training_index, output, device="cuda:0"):
    index_path = Path(training_index).resolve()
    index_digest = sha256(index_path)
    protocol, index = read_json(PROTOCOL), read_json(index_path)
    revision = git_info()
    if revision != index.get("git"):
        raise ValueError("inference requires the exact clean training Git revision and branch")
    verify_inputs(index, protocol)
    rows = build_rows(protocol)
    output = resolve_run_dir(Path(output)).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output == ROOT or ROOT in output.parents:
        raise ValueError("inference output must be outside the source checkout")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "generation_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    sampling = protocol["inference"]
    provenance = {"status": "running", "git": revision, "protocol_sha256": protocol_hash(protocol),
                  "training_index": str(index_path), "training_index_sha256": index_digest,
                  "training": index["runs"], "sampling": sampling, "row_count": len(rows),
                  "environment": package_versions(), "device": device,
                  "internal_gate_labels": {"subject": "<T*>", "material": "<M*>"},
                  "literal_baseline": "Both adapters installed; both learned-token gates exactly zero."}
    write_json(output / "provenance.json", provenance)
    generated = []
    try:
        import torch
        for method, texture, material in itertools.product(
                METHODS, target_ids(protocol, "texture"), target_ids(protocol, "material")):
            selections = [next(item for item in index["runs"] if
                               (item["method"], item["branch"], item["concept_id"]) == (method, branch, concept))
                          for branch, concept in (("texture", texture), ("material", material))]
            pipe = load_pipeline(selections, device)
            with (output / "generation_status.jsonl").open("a", encoding="utf-8") as ledger, torch.no_grad():
                for row in rows:
                    if (row["method"], row["texture"], row["material"]) != (method, texture, material):
                        continue
                    result = pipe(row["prompt"], num_inference_steps=sampling["num_inference_steps"],
                                  guidance_scale=sampling["guidance_scale"], width=sampling["resolution"],
                                  height=sampling["resolution"],
                                  generator=torch.Generator(device=device).manual_seed(row["seed"]),
                                  cross_attention_kwargs=attention_kwargs(pipe, row["prompt"], sampling["guidance_scale"]))
                    flags = result.nsfw_content_detected
                    if flags is None or len(flags) != 1:
                        raise ValueError("missing safety checker result")
                    path = output / row["image_path"]
                    result.images[0].save(path)
                    record = {**row, "status": "safety_filtered" if bool(flags[0]) else "ok",
                              "image_sha256": sha256(path)}
                    generated.append(record)
                    ledger.write(json.dumps(record) + "\n")
                    ledger.flush()
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
        verify_inputs(index, protocol)
        if sha256(index_path) != index_digest or protocol_hash(read_json(PROTOCOL)) != protocol_hash(protocol):
            raise ValueError("training index or protocol changed during inference")
        if git_info() != revision:
            raise ValueError("Git state changed during inference")
        if len(generated) != len(rows) or {r["id"] for r in generated} != {r["id"] for r in rows}:
            raise ValueError("generation ledger is incomplete")
        sheets = review_bundle(output, generated, protocol)
        write_json(output / "summary.json", {
            "status": "succeeded", "expected_images": len(rows), "generated_images": len(generated),
            "missing_images": 0, "safety_filtered": sum(r["status"] == "safety_filtered" for r in generated),
            "by_method": {m: sum(r["method"] == m for r in generated) for m in METHODS},
            "contact_sheets": sheets, "human_ratings": "pending", "phase2_started": False,
            "limitations": ["A 300-step synthetic pilot establishes execution, not successful disentanglement.",
                            "Token-local gating restricts text positions; semantic independence requires evaluation.",
                            "No BRDF, texture accuracy or leakage score is inferred from generation counts.",
                            "Describe the six observed attributes blindly, then join the key to assess adherence and leakage.",
                            "Literal arms use the same dual-adapter pipeline with zero learned-token gates."]})
        provenance["status"] = "succeeded"
    except Exception as error:
        provenance.update(status="failed", generated_images=len(generated), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "provenance.json", provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.training_index, args.output, args.device)


if __name__ == "__main__":
    main()
