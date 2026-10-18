"""Probe native/literal/learned Texture without any Material prompt or adapter."""

import argparse
from collections import Counter
import gc
import itertools
import json
from pathlib import Path

from experiments.sd15_token_local_kv_v1.evaluate import attention_masks, load_pipeline
from experiments.sd_native_comparison_v1.generate import package_versions, write_json
from experiments.texture_material_disentangled_v1 import data, infer
from experiments.texture_material_disentangled_v1.texture_probe_source import verify_source
from experiments.texture_material_disentangled_v1.texture_probe_review import review_bundle
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
STEPS = (500, 750, 1000)
CONDITIONS = ("base", "literal", "learned_texture")


def build_rows(protocol, step):
    if step not in STEPS:
        raise ValueError("select an existing500/750/1000-step snapshot")
    rows = []
    for method, texture, context, seed, condition in itertools.product(
            infer.METHODS, infer.target_ids(protocol, "texture"), protocol["dataset"]["heldout_contexts"],
            protocol["inference"]["seeds"], CONDITIONS):
        prompt = f"a photo of a {context['color']} {context['shape']}"
        if condition != "base":
            token = "<T*>" if condition == "learned_texture" else protocol["textures"][texture]["literal"]
            prompt += f" with {token} texture"
        prompt += f", {protocol['dataset']['light_phrases'][context['light']]}"
        identity = f"{method}__{texture}__{context['id']}__{condition}__seed{seed}"
        rows.append({"id": identity, "method": method, "texture": texture, "context": context["id"],
                     "shape": context["shape"], "color": context["color"], "light": context["light"],
                     "seed": seed, "condition": condition, "training_steps": step,
                     "prompt": prompt, "image_path": f"images/{identity}.png"})
    if len(rows) != 96 or len({row["id"] for row in rows}) != 96:
        raise ValueError("expected96 unique matched Texture probe rows")
    return rows


def generate(pipe, rows, output, sampling, device, learned):
    import torch
    with (output / "generation_status.jsonl").open("a", encoding="utf-8") as ledger, torch.no_grad():
        for row in rows:
            kwargs = {"cross_attention_kwargs": attention_masks(pipe, row["prompt"], ["<T*>"],
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
            ledger.write(json.dumps({**row, "status": "safety_filtered" if bool(flags[0]) else "ok",
                "image_sha256": data.sha256(path), "texture_adapter_loaded": learned,
                "material_adapter_loaded": False}) + "\n")
            ledger.flush()


def run(source, output, step, device="cuda:0"):
    source, output = Path(source).resolve(), resolve_run_dir(Path(output))
    if output.exists() or output.is_relative_to(source) or output.is_relative_to(ROOT):
        raise ValueError("Texture probe requires a new external output outside the source experiment")
    revision, protocol = git_info(), data.read_protocol()
    index, selections, locks, references = verify_source(source, step)
    rows = build_rows(protocol, step)
    sampling = {**protocol["inference"], "conditions": list(CONDITIONS)}
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "generation_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    provenance = {"status": "running", "git": revision, "training_git": index["training_git"],
        "source_export_git": index["git"], "source_run": str(source), "source_locks": locks,
        "training_steps": step, "training": selections, "sampling": sampling,
        "protocol_sha256": data.protocol_sha256(), "environment": package_versions(), "device": device,
        "row_count": len(rows), "conditions": list(CONDITIONS), "material_adapter_loaded": False,
        "base_and_literal": "Native SD1.5; no concept adapter or learned embedding.",
        "learned_texture": "Only selected Texture adapter/embedding; no Material adapter.",
        "training_images": "training_references.json; all16 actual staged RGB images per learned concept",
        "retraining": False}
    write_json(output / "provenance.json", provenance)
    try:
        import torch
        for selection in selections:
            pipe = load_pipeline([selection], device)
            generate(pipe, [row for row in rows if row["condition"] == "learned_texture"
                     and (row["method"], row["texture"]) == (selection["method"], selection["concept_id"])],
                     output, sampling, device, True)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
        pipe = load_pipeline([], device)
        generate(pipe, [row for row in rows if row["condition"] != "learned_texture"], output, sampling, device, False)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        generated = [json.loads(line) for line in (output / "generation_status.jsonl").read_text().splitlines()]
        expected = {row["id"]: row for row in rows}
        if len(generated) != len(rows) or {row["id"] for row in generated} != set(expected):
            raise ValueError("incomplete or duplicated Texture probe ledger")
        for row in generated:
            if (any(row.get(key) != value for key, value in expected[row["id"]].items())
                    or row.get("material_adapter_loaded") is not False
                    or row.get("texture_adapter_loaded") != (row["condition"] == "learned_texture")
                    or data.sha256(output / row["image_path"]) != row["image_sha256"]):
                raise ValueError("Texture generation condition or image hash differs")
        verify_source(source, step)
        if (any(data.sha256(Path(path)) != digest for path, digest in locks.items())
                or data.read_protocol() != protocol or git_info() != revision):
            raise ValueError("source checkpoint, training images or checkout changed")
        sheets = review_bundle(output, generated, protocol, references)
        write_json(output / "summary.json", {"status": "succeeded", "training_steps": step,
            "expected_images": len(rows), "generated_images": len(generated), "missing_images": 0,
            "by_method": dict(Counter(row["method"] for row in generated)),
            "by_condition": dict(Counter(row["condition"] for row in generated)),
            "safety_filtered": sum(row["status"] == "safety_filtered" for row in generated),
            "contact_sheets": sheets, "training_references": "training_references.json",
            "human_ratings": "pending", "material_leakage_score": None,
            "limitations": ["Omitting a Material prompt does not require a material-free image.",
                "Compare matched base/literal/learned arms to assess changes in reflectance appearance.",
                "Training references show the actual varied-material RGB supervision, not inferred texture maps."]})
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
    parser.add_argument("--step", choices=STEPS, type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.source_run, args.output, args.step, args.device)
