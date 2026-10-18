"""Natural T/M recipes with the established Phase 1 training arguments."""

from pathlib import Path

from experiments.natural_texture_material_v1 import dataset_protocol as schema
from experiments.texture_material_disentangled_v1 import step_ablation

STEPS = (500, 750, 1000)


def recipes():
    return [{"method": method, "branch": branch, "concept_id": source + "_" + branch,
             "source_id": source, "token": "<T*>" if branch == "texture" else "<M*>"}
            for source in schema.SOURCES for branch in ("texture", "material")
            for method in ("fullrank", "lora_r64")]


def _template(recipe):
    if recipe not in recipes():
        raise ValueError("expected one of the twelve natural Texture/Material recipes")
    return {key: value for key, value in {**recipe,
        "concept_id": "vertical_stripes" if recipe["branch"] == "texture" else "polished_metal"}.items()
        if key != "source_id"}


def build_config(recipe, staging, protocol):
    baseline = step_ablation.build_config(_template(recipe), staging)
    concept = Path(staging).resolve() / recipe["concept_id"]
    return {**baseline,
        "run": {"study": "natural_texture_material_v1",
                "variant": f"{recipe['concept_id']}_{recipe['method']}_steps1000", "seed": 42},
        "data_manifest": str(concept / "training_assets_manifest.jsonl"),
        "texture_material": {**recipe, "mode": "pilot", "protocol_sha256": schema.protocol_sha256(protocol)},
        "args": {**baseline["args"], "concepts_list": str(concept / "concepts.json"),
                 "checkpoint_steps": list(STEPS)}}


def compare_config(recipe, config, staging, protocol):
    template = _template(recipe)
    baseline = step_ablation.build_config(template, staging)
    expected = build_config(recipe, staging, protocol)
    if config != expected:
        raise ValueError("natural training config differs from the frozen recipe")
    differences = {key: {"old": baseline["args"].get(key), "new": config["args"].get(key)}
                   for key in set(baseline["args"]) | set(config["args"])
                   if baseline["args"].get(key) != config["args"].get(key)}
    if set(differences) != {"concepts_list", "checkpoint_steps"}:
        raise ValueError("only concept data and explicit checkpoint steps may change")
    return {"recipe": recipe, "phase1_template_recipe": template,
            "argument_differences_at_same_staging": differences,
            "other_training_arguments_unchanged": True}
