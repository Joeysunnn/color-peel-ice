"""Read-only source locks for the wood Material initializer ablation."""

import copy
import json
from pathlib import Path

from experiments.natural_texture_material_v1 import dataset_protocol as schema
from experiments.natural_texture_material_v1.probe import STEPS, TRANSFER_PROMPTS, build_rows
from experiments.natural_texture_material_v1.probe_source import audit_training
from experiments.natural_texture_material_v1.training_config import recipes as natural_recipes
from scripts.launch.colorpeel_run import read_config


def build_config(baseline, method):
    args = baseline["args"]
    if (method not in ("fullrank", "lora_r64") or args.get("modifier_token") != "<M*>"
            or args.get("initializer_token") != "metal" or args.get("max_train_steps") != 1000
            or args.get("checkpoint_steps") != list(STEPS)
            or baseline.get("texture_material", {}).get("concept_id") != "wood_spoon_material"
            or baseline["texture_material"].get("method") != method
            or (method == "fullrank" and args.get("token_local_kv") is not True)
            or (method == "lora_r64" and (args.get("subject_lora_mode") != "token_local_kv"
                or args.get("subject_lora_rank") != 64 or args.get("subject_lora_alpha") != 64
                or args.get("token_local_kv", False)))):
        raise ValueError("Expected the original wood Material method and 1000-step recipe")
    config = copy.deepcopy(baseline)
    config["args"]["initializer_token"] = "material"
    config["run"].update(study="natural_material_initializer_v1",
        variant=f"wood_spoon_material_{method}_init_material_steps1000")
    return config


def _baseline_rows(folder, expected, locks):
    ledger = folder / "generation_status.jsonl"
    locks[str(ledger)] = schema.sha256(ledger)
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf8").splitlines()]
    by_id = {row["id"]: row for row in rows}
    if len(rows) != 288 or len(by_id) != 288:
        raise ValueError("Expected 288 unique completed baseline generation rows")
    if set(by_id) != {row["id"] for row in expected}:
        raise ValueError("Baseline generation identities differ")
    selected = []
    for reference in expected:
        if reference["concept_id"] != "wood_spoon_material":
            continue
        row = by_id[reference["id"]]
        learned = row["condition"] == "learned_material"
        image = (folder / row["image_path"]).resolve()
        if (any(row.get(key) != value for key, value in reference.items())
                or row.get("status") not in ("ok", "safety_filtered")
                or row.get("texture_adapter_loaded") is not False
                or row.get("material_adapter_loaded") is not learned
                or schema.sha256(image) != row.get("image_sha256")):
            raise ValueError("Baseline image, prompt, status or loaded adapter differs")
        locks[str(image)] = row["image_sha256"]
        selected.append({**row, "source_image": str(image)})
    if len(selected) != 48:
        raise ValueError("Expected 48 wood Material baseline rows per step")
    return selected


def prepare(source_training, source_transfer):
    training, transfer = Path(source_training).resolve(), Path(source_transfer).resolve()
    audit = audit_training(training)
    locks = dict(audit["source_locks"])

    def read(path):
        path = Path(path)
        digest = schema.sha256(path)
        if str(path) in locks and locks[str(path)] != digest:
            raise ValueError("Frozen source changed while preparing initializer study")
        locks[str(path)] = digest
        return json.loads(path.read_text(encoding="utf8"))

    state = read(transfer / "status.json")
    if (state.get("status") != "execution_verified" or state.get("generated_images") != 864
            or state.get("expected_images") != 864 or state.get("source_run") != str(training)
            or state.get("probe_suite") != "ordinary_object_transfer"
            or state.get("training_git") != audit["training_git"]):
        raise ValueError("Expected the completed ordinary-object transfer from the original training")
    prompts = read(TRANSFER_PROMPTS)
    if (read(transfer / "transfer_protocol.json") != prompts
            or state.get("prompt_protocol_sha256") != schema.sha256(TRANSFER_PROMPTS)):
        raise ValueError("Baseline transfer prompts differ from the accepted ordinary prompts")
    protocol = read(training / "protocol.json")
    recipes = [item for item in natural_recipes() if item["concept_id"] == "wood_spoon_material"]
    parents = read(training / "training/training_index.json")["runs"]
    configs = {item["method"]: read_config(Path(item["run"]) / "config.yaml")
               for item in parents if item["concept_id"] == "wood_spoon_material"}
    for method, config in configs.items():
        build_config(config, method)
    group = read(training / "training_references.json")["groups"]["wood_spoon_material"]
    ledger = Path(group["training_assets_manifest"])
    actual = [json.loads(line) for line in ledger.read_text(encoding="utf8").splitlines()]
    sampling = {**protocol["inference"], "conditions": {
        "texture": ["base", "literal", "learned_texture"],
        "material": ["base", "literal", "learned_material"]}}
    references, baseline_rows = {}, {}
    for step in STEPS:
        folder = transfer / f"step{step}"
        summary, provenance = read(folder / "summary.json"), read(folder / "provenance.json")
        selections = read(training / f"step{step}/training_index.json")["runs"]
        if (summary.get("status") != "succeeded" or summary.get("generated_images") != 288
                or summary.get("missing_images") != 0 or summary.get("training_steps") != step
                or provenance.get("status") != "succeeded" or provenance.get("source_run") != str(training)
                or provenance.get("training_steps") != step or provenance.get("row_count") != 288
                or provenance.get("probe_suite") != "ordinary_object_transfer"
                or provenance.get("git") != state.get("git")
                or provenance.get("training_git") != audit["training_git"]
                or provenance.get("sampling") != sampling or provenance.get("training") != selections
                or provenance.get("prompt_protocol") != prompts
                or provenance.get("prompt_protocol_sha256") != schema.sha256(TRANSFER_PROMPTS)
                or provenance.get("protocol_sha256") != schema.protocol_sha256(protocol)):
            raise ValueError("Baseline transfer completion, sampling or checkpoint provenance differs")
        reference_groups = read(folder / "training_references.json")["groups"]
        chosen_refs = {}
        for item in selections:
            if item["concept_id"] != "wood_spoon_material":
                continue
            expected = {"method": item["method"], "branch": "material", "source_id": "wood_spoon",
                "concept_id": "wood_spoon_material", "step": step, "run": item["run"],
                "checkpoint": item["checkpoint"], "source": group["source"],
                "dataset_manifest": str(ledger), "dataset_manifest_sha256": schema.sha256(ledger),
                "images": [{key: row[key] for key in ("id", "image", "image_sha256", "prompt", "shape",
                    "color", "light", "material_id", "roughness", "metallic", "seed") if key in row}
                    for row in actual], "conditions": {"base": None, "literal": None, "learned_material": "images"}}
            matching = [ref for ref in reference_groups if ref.get("method") == item["method"]
                        and ref.get("concept_id") == "wood_spoon_material"]
            if matching != [expected] or len(actual) != 8:
                raise ValueError("Baseline references do not match the actual eight wood training RGB images")
            chosen_refs[item["method"]] = expected
        references[str(step)] = chosen_refs
        baseline_rows[str(step)] = _baseline_rows(folder, build_rows(protocol, step, True), locks)
    if any(schema.sha256(path) != digest for path, digest in locks.items()):
        raise ValueError("Training or baseline sources changed while preparing the experiment")
    return {"protocol": protocol, "recipes": recipes, "configs": configs, "references": references,
            "baseline_rows": baseline_rows, "source_locks": locks,
            "training_git": audit["training_git"], "transfer_git": state["git"]}
