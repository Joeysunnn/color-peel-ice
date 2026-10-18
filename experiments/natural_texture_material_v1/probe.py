"""Matched native/literal/learned natural T/M probes, with no opposite-branch prompt."""

import argparse
from collections import Counter
import gc
import itertools
import json
from pathlib import Path

from experiments.sd15_token_local_kv_v1.evaluate import attention_masks, load_pipeline
from experiments.sd_native_comparison_v1.generate import package_versions
from experiments.natural_texture_material_v1 import dataset_protocol as schema
from experiments.natural_texture_material_v1.probe_source import verify_source
from experiments.natural_texture_material_v1.probe_review import review_bundle
from scripts.launch.colorpeel_run import git_info, resolve_run_dir

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = Path(__file__).with_name("probe_protocol.json")
TRANSFER_PROMPTS = Path(__file__).with_name("transfer_protocol.json")
STEPS = (500, 750, 1000)
METHODS = ("fullrank", "lora_r64")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def prompt_contexts(protocol, transfer=False):
    if transfer:
        return [{"id": name, "shape": name, "color": None, "light": None}
                for name in json.loads(TRANSFER_PROMPTS.read_text())["objects"]]
    return protocol["dataset"]["heldout_contexts"]


def build_rows(protocol, step, transfer=False):
    if step not in STEPS:
        raise ValueError("Select an existing500/750/1000-step snapshot")
    literals, rows = json.loads((TRANSFER_PROMPTS if transfer else PROMPTS).read_text()), []
    for method, branch, source, context, seed in itertools.product(
            METHODS, ("texture", "material"), schema.SOURCES,
            prompt_contexts(protocol, transfer), protocol["inference"]["seeds"]):
        for condition in ("base", "literal", "learned_" + branch):
            prompt = (literals["base_template"].format(object=context["shape"]) if transfer
                      else f"a photo of a {context['color']} {context['shape']}")
            if condition != "base":
                token = ("<T*>" if branch == "texture" else "<M*>") if condition.startswith("learned_") else literals["literal_" + branch][source]
                if transfer:
                    prompt = literals[branch + "_template"].format(object=context["shape"], attribute=token)
                else:
                    prompt += f" with {token} texture" if branch == "texture" else f" made of {token}"
            if not transfer:
                prompt += f", {protocol['dataset']['light_phrases'][context['light']]}"
            identity = f"{method}__{source}_{branch}__{context['id']}__{condition}__seed{seed}"
            rows.append({"id": identity, "method": method, "branch": branch, "source_id": source,
                         "concept_id": source + "_" + branch, "context": context["id"],
                         "shape": context["shape"], "color": context["color"], "light": context["light"],
                         "seed": seed, "condition": condition, "training_steps": step,
                         "prompt": prompt, "image_path": f"images/{identity}.png"})
    if len(rows) != 288 or len({row["id"] for row in rows}) != 288:
        raise ValueError("Expected288 unique matched natural probe rows")
    return rows


def generate(pipe, rows, output, sampling, device, learned):
    import torch
    with (output / "generation_status.jsonl").open("a", encoding="utf8") as ledger, torch.no_grad():
        for row in rows:
            token = "<T*>" if row["branch"] == "texture" else "<M*>"
            kwargs = {"cross_attention_kwargs": attention_masks(pipe, row["prompt"], [token],
                       sampling["guidance_scale"])} if learned else {}
            result = pipe(row["prompt"], num_inference_steps=sampling["num_inference_steps"],
                          guidance_scale=sampling["guidance_scale"], width=sampling["resolution"],
                          height=sampling["resolution"],
                          generator=torch.Generator(device=device).manual_seed(row["seed"]), **kwargs)
            flags = result.nsfw_content_detected
            if flags is None or len(flags) != 1:
                raise ValueError("Missing safety checker result")
            path = output / row["image_path"]
            result.images[0].save(path)
            ledger.write(json.dumps({**row, "status": "safety_filtered" if bool(flags[0]) else "ok",
                "image_sha256": schema.sha256(path),
                "texture_adapter_loaded": learned and row["branch"] == "texture",
                "material_adapter_loaded": learned and row["branch"] == "material"}) + "\n")
            ledger.flush()


def run(source, output, step, device="cuda:0", training_audit=None, transfer=False):
    source, output = Path(source).resolve(), resolve_run_dir(Path(output))
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(source):
        raise ValueError("Use a new external probe directory outside training artifacts")
    prompts = TRANSFER_PROMPTS if transfer else PROMPTS
    git, prompt_hash = git_info(), schema.sha256(prompts)
    audit = json.loads(Path(training_audit).read_text()) if training_audit else None
    protocol, selections, locks, references = verify_source(source, step, audit)
    rows = build_rows(protocol, step, transfer)
    sampling = {**protocol["inference"], "conditions": {
        "texture": ["base", "literal", "learned_texture"],
        "material": ["base", "literal", "learned_material"]}}
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "generation_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    provenance = {"status": "running", "git": git, "training_git": json.loads((source / "status.json").read_text())["git"],
        "source_run": str(source), "source_locks": locks, "training_steps": step,
        "training": selections, "sampling": sampling, "protocol_sha256": schema.protocol_sha256(protocol),
        "probe_suite": "ordinary_object_transfer" if transfer else "neutral_geometry",
        "prompt_protocol_sha256": prompt_hash, "prompt_protocol": json.loads(prompts.read_text()),
        "environment": package_versions(), "device": device, "row_count": 288, "retraining": False,
        "base_and_literal": "Native SD1.5; no concept adapter or learned embedding.",
        "learned": "Only the selected branch's single adapter and embedding; opposite adapter absent.",
        "training_references": "training_references.json; actual16Texture/8Material RGB images per group"}
    write(output / "provenance.json", provenance)
    try:
        import torch
        for selection in selections:
            pipe = load_pipeline([selection], device)
            chosen = [row for row in rows if row["condition"].startswith("learned_")
                      and (row["method"], row["concept_id"]) == (selection["method"], selection["concept_id"])]
            generate(pipe, chosen, output, sampling, device, True)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
        pipe = load_pipeline([], device)
        generate(pipe, [row for row in rows if not row["condition"].startswith("learned_")],
                 output, sampling, device, False)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        generated = [json.loads(line) for line in (output / "generation_status.jsonl").read_text().splitlines()]
        expected = {row["id"]: row for row in rows}
        if len(generated) != 288 or {row["id"] for row in generated} != set(expected):
            raise ValueError("Incomplete or duplicated generation ledger")
        for row in generated:
            learned = row["condition"].startswith("learned_")
            if (any(row.get(key) != value for key, value in expected[row["id"]].items())
                    or row.get("texture_adapter_loaded") != (learned and row["branch"] == "texture")
                    or row.get("material_adapter_loaded") != (learned and row["branch"] == "material")
                    or schema.sha256(output / row["image_path"]) != row["image_sha256"]):
                raise ValueError("Image hash or branch/adapter condition differs")
        verify_source(source, step, audit)
        if (git_info() != git or schema.sha256(prompts) != prompt_hash
                or any(schema.sha256(path) != digest for path, digest in locks.items())):
            raise ValueError("Source checkpoint, actual training images, prompts or checkout changed")
        sheets = review_bundle(output, generated, protocol, references,
                               contexts=prompt_contexts(protocol, transfer))
        write(output / "summary.json", {"status": "succeeded", "training_steps": step,
            "generated_images": 288, "expected_images": 288, "missing_images": 0,
            "by_method": dict(Counter(row["method"] for row in generated)),
            "by_branch": dict(Counter(row["branch"] for row in generated)),
            "by_condition": dict(Counter(row["condition"] for row in generated)),
            "safety_filtered": sum(row["status"] == "safety_filtered" for row in generated),
            "contact_sheets": sheets, "human_ratings": "pending", "disentanglement_score": None,
            "limitations": [("Ordinary object prompts test transfer; visual attribute independence still needs review."
                             if transfer else "Neutral prompts do not require appearance without material or texture."),
                "Literal words are semantic controls, not exact descriptions of extracted R/M or spatial patterns.",
                "Matched visual assessment is needed; execution success does not establish independence."]})
        provenance["status"] = "succeeded"
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output / "provenance.json", provenance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=STEPS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-audit", type=Path)
    parser.add_argument("--transfer", action="store_true", help="Test chair/mug/vase/sofa ordinary prompts")
    args = parser.parse_args()
    run(args.source_run, args.output, args.step, args.device, args.training_audit, args.transfer)
