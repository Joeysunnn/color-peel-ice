"""Compare native, literal and learned Material without any texture prompt."""

import argparse
from collections import Counter
import csv
import gc
import html
import itertools
import json
from pathlib import Path
import random

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd15_token_local_kv_v1.evaluate import attention_masks, load_pipeline
from experiments.sd_native_comparison_v1.generate import package_versions, write_json
from experiments.texture_material_disentangled_v1 import data, infer
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
STEPS = (500, 750, 1000)
CONDITIONS = ("base", "literal", "learned_material")


def build_rows(protocol, step):
    if step not in STEPS:
        raise ValueError("select an existing500/750/1000-step snapshot")
    rows = []
    for method, material, context, seed, condition in itertools.product(
            infer.METHODS, infer.target_ids(protocol, "material"),
            protocol["dataset"]["heldout_contexts"], protocol["inference"]["seeds"], CONDITIONS):
        prompt = f"a photo of a {context['color']} {context['shape']}"
        if condition != "base":
            token = "<M*>" if condition == "learned_material" else protocol["materials"][material]["literal"]
            prompt += f" made of {token}"
        prompt += f", {protocol['dataset']['light_phrases'][context['light']]}"
        identity = f"{method}__{material}__{context['id']}__{condition}__seed{seed}"
        rows.append({"id": identity, "method": method, "material": material,
                     "context": context["id"], "shape": context["shape"], "color": context["color"],
                     "light": context["light"], "seed": seed, "condition": condition,
                     "training_steps": step, "prompt": prompt, "image_path": f"images/{identity}.png"})
    if len(rows) != 96 or len({row["id"] for row in rows}) != 96:
        raise ValueError("expected the fixed96-image three-arm Material matrix")
    return rows


def verify_source(source, step):
    """Read the completed step experiment; retain its original training provenance."""
    if step not in STEPS or read_json(source / "status.json").get("status") != "execution_verified":
        raise ValueError("expected the completed source step experiment")
    index_path = source / f"step{step}/training_index.json"
    index = read_json(index_path)
    protocol = data.read_protocol()
    infer.verify_inputs(index, protocol, step)
    provenance_path = source / f"step{step}/inference/provenance.json"
    provenance = read_json(provenance_path)
    if (index.get("training_steps") != step or index.get("source_training_steps") != 1000
            or provenance.get("status") != "succeeded" or provenance.get("training_steps") != step
            or provenance.get("sampling") != protocol["inference"]
            or provenance.get("training_index_sha256") != sha256(index_path)):
        raise ValueError("source step or sampling provenance differs")
    selections = [item for item in index["runs"] if item["branch"] == "material"]
    for item in selections:
        config = read_json(Path(item["checkpoint"]) / "adaptation_config.json")
        mode = "token_local_kv" if item["method"] == "fullrank" else "lora_material_kv"
        if (config.get("modifier_tokens") != ["<M*>"] or config.get("adaptation_mode") != mode
                or (item["method"] == "lora_r64" and any(config.get(k) != v for k, v in
                    {"rank": 64, "alpha": 64, "mode": "token_local_kv"}.items()))):
            raise ValueError("Material adapter token, mode or rank differs")
    locks = {str(path): sha256(path) for path in (source / "status.json", index_path, provenance_path)}
    return index, selections, locks


def generate(pipe, rows, output, sampling, device, learned):
    import torch
    with (output / "generation_status.jsonl").open("a", encoding="utf-8") as ledger, torch.no_grad():
        for row in rows:
            kwargs = {"cross_attention_kwargs": attention_masks(pipe, row["prompt"], ["<M*>"],
                       sampling["guidance_scale"])} if learned else {}
            result = pipe(row["prompt"], num_inference_steps=sampling["num_inference_steps"],
                          guidance_scale=sampling["guidance_scale"], width=sampling["resolution"],
                          height=sampling["resolution"],
                          generator=torch.Generator(device=device).manual_seed(row["seed"]), **kwargs)
            flags = result.nsfw_content_detected
            if flags is None or len(flags) != 1:
                raise ValueError("missing safety checker result")
            path = output / row["image_path"]
            result.images[0].save(path)
            record = {**row, "status": "safety_filtered" if bool(flags[0]) else "ok",
                      "image_sha256": sha256(path), "material_adapter_loaded": learned}
            ledger.write(json.dumps(record) + "\n")
            ledger.flush()


def review_bundle(output, rows, protocol):
    from PIL import Image, ImageDraw
    sheets = []
    (output / "contact_sheets").mkdir()
    by_key = {(r["method"], r["material"], r["context"], r["seed"], r["condition"]): r for r in rows}
    for method, material, seed in itertools.product(infer.METHODS, infer.target_ids(protocol, "material"),
                                                   protocol["inference"]["seeds"]):
        sheet = Image.new("RGB", (3 * 272, 56 + 4 * 294), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"{method} / {material} / seed{seed} / step{rows[0]['training_steps']}", fill="black")
        for column, condition in enumerate(CONDITIONS):
            draw.text((8 + 272 * column, 32), condition, fill="black")
        for line, context in enumerate(protocol["dataset"]["heldout_contexts"]):
            for column, condition in enumerate(CONDITIONS):
                row = by_key[method, material, context["id"], seed, condition]
                x, y = 8 + 272 * column, 56 + 294 * line
                with Image.open(output / row["image_path"]) as image:
                    sheet.paste(image.convert("RGB").resize((256, 256)), (x, y))
                draw.text((x, y + 258), context["id"], fill="black")
                if row["status"] == "safety_filtered":
                    draw.text((x, y + 272), "SAFETY FILTERED", fill="red")
        path = f"contact_sheets/{method}__{material}__seed{seed}.jpg"
        sheet.save(output / path, quality=90)
        sheets.append(path)
    shuffled = list(rows)
    random.Random(protocol["seed"]).shuffle(shuffled)
    key, cards = [], []
    with (output / "blinded_ratings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_id", "material_appearance", "extra_patterns",
                                                     "shape", "color", "lighting", "background", "notes"])
        writer.writeheader()
        for index, row in enumerate(shuffled):
            blind_id = f"sample_{index:04d}"
            key.append({"blind_id": blind_id, **row})
            if row["status"] != "safety_filtered":
                writer.writerow({"blind_id": blind_id})
                cards.append(f'<figure><img width="256" src="{html.escape(row["image_path"], quote=True)}">'
                             f'<figcaption>{blind_id}</figcaption></figure>')
    write_json(output / "blinded_key.json", {"samples": key})
    (output / "blinded_review.html").write_text('<!doctype html><meta charset="utf-8"><title>Material review</title>'
        '<p>Describe visible material and patterns in blinded_ratings.csv before opening blinded_key.json.</p>'
        + "".join(cards), encoding="utf-8")
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Material probe</title>'
        '<p>Columns: native SD1.5, literal material, learned M. No texture prompts or Texture adapter.</p>'
        '<p>Images support visual assessment; no physical BRDF or leakage score has been inferred.</p>'
        '<a href="blinded_review.html">Blinded review</a>' + "".join(
            f'<p><a href="{path}">{path}</a></p>' for path in sheets), encoding="utf-8")
    return sheets


def run(source, output, step, device="cuda:0", uniform_material=False):
    source, output = Path(source).resolve(), resolve_run_dir(Path(output))
    if output.exists() or output.is_relative_to(source) or output.is_relative_to(ROOT):
        raise ValueError("probe output must be a new external directory outside the source experiment")
    revision, protocol = git_info(), data.read_protocol()
    if uniform_material:
        from experiments.material_uniform_albedo_v1.audit import verify_source as source_verifier
    else:
        source_verifier = verify_source
    index, selections, locks = source_verifier(source, step)
    rows = build_rows(protocol, step)
    sampling = {**protocol["inference"], "conditions": list(CONDITIONS)}
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "generation_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    provenance = {"status": "running", "git": revision, "training_git": index["training_git"],
                  "source_inference_git": None if uniform_material else index["git"],
                  "source_export_git": index["git"], "source_run": str(source), "source_locks": locks,
                  "source_kind": "uniform_material" if uniform_material else "original_material",
                  "training_protocol_sha256": index.get("protocol_sha256"),
                  "training_steps": step, "training": selections, "sampling": sampling,
                  "protocol_sha256": data.protocol_sha256(), "environment": package_versions(),
                  "row_count": len(rows), "conditions": list(CONDITIONS), "texture_adapter_loaded": False,
                  "base_and_literal": "Native pipeline without adapters or learned embeddings.",
                  "learned_material": "Only the selected M adapter/embedding; single Boolean token-position gate.",
                  "device": device, "retraining": False}
    write_json(output / "provenance.json", provenance)
    try:
        import torch
        for selection in selections:
            pipe = load_pipeline([selection], device)
            generate(pipe, [row for row in rows if row["condition"] == "learned_material"
                     and (row["method"], row["material"]) == (selection["method"], selection["concept_id"])],
                     output, sampling, device, True)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
        pipe = load_pipeline([], device)
        generate(pipe, [row for row in rows if row["condition"] != "learned_material"], output, sampling, device, False)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        generated = [json.loads(line) for line in (output / "generation_status.jsonl").read_text().splitlines()]
        expected = {row["id"]: row for row in rows}
        if len(generated) != len(rows) or {row["id"] for row in generated} != set(expected):
            raise ValueError("incomplete or duplicated generation ledger")
        for row in generated:
            if (any(row.get(key) != value for key, value in expected[row["id"]].items())
                    or sha256(output / row["image_path"]) != row["image_sha256"]):
                raise ValueError("generation condition or image changed")
        if uniform_material:
            source_verifier(source, step)
        else:
            infer.verify_inputs(index, protocol, step)
        if (any(sha256(Path(path)) != digest for path, digest in locks.items())
                or data.read_protocol() != protocol or git_info() != revision):
            raise ValueError("source artifacts, protocol or checkout changed")
        sheets = review_bundle(output, generated, protocol)
        write_json(output / "summary.json", {"status": "succeeded", "training_steps": step,
            "expected_images": len(rows), "generated_images": len(generated), "missing_images": 0,
            "by_method": dict(Counter(row["method"] for row in generated)),
            "by_condition": dict(Counter(row["condition"] for row in generated)),
            "safety_filtered": sum(row["status"] == "safety_filtered" for row in generated),
            "contact_sheets": sheets, "human_ratings": "pending", "texture_leakage_score": None,
            "limitations": ["Neutral prompts do not require a perfectly uniform surface.",
                "Compare matched base/literal/learned arms; visible highlights are not automatically albedo patterns.",
                "Generation counts establish execution, not successful disentanglement."]})
        provenance["status"] = "succeeded"
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "provenance.json", provenance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=STEPS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--uniform-material", action="store_true",
                        help="Use audited uniform-albedo Material exports; keep the original probe prompts/sampling.")
    args = parser.parse_args()
    run(args.source_run, args.output, args.step, args.device, args.uniform_material)
