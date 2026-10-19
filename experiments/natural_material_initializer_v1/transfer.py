"""Generate only the new-token arm and preserve matched old image controls."""

import argparse
import gc
import json
from pathlib import Path
import shutil

from experiments.natural_material_initializer_v1.review import review_bundle
from experiments.natural_material_initializer_v1.run import verify_locks
from experiments.natural_material_initializer_v1.source import SOURCE_IDS
from experiments.natural_texture_material_v1 import probe
from experiments.texture_material_disentangled_v1 import step_snapshots
from scripts.launch.colorpeel_run import git_info


def comparison_rows(protocol, baseline, step, source_id="wood_spoon"):
    if source_id not in SOURCE_IDS or any(
            row["source_id"] != source_id or row["concept_id"] != source_id + "_material"
            or row["branch"] != "material" for row in baseline):
        raise ValueError("Initializer comparison must contain one selected Material source")
    rows = []
    for old in baseline:
        condition = "learned_metal_init" if old["condition"] == "learned_material" else old["condition"]
        identity = old["id"].replace("__learned_material__", "__learned_metal_init__")
        rows.append({**old, "condition": condition, "id": identity,
                     "image_path": f"images/{identity}.png", "reused": True})
    for row in probe.build_rows(protocol, step, transfer=True):
        if row["concept_id"] != source_id + "_material" or row["condition"] != "learned_material":
            continue
        identity = row["id"].replace("__learned_material__", "__learned_material_init__")
        rows.append({**row, "condition": "learned_material_init", "id": identity,
                     "image_path": f"images/{identity}.png", "reused": False})
    if len(rows) != 64 or len({row["id"] for row in rows}) != 64:
        raise ValueError("Expected64 matched initializer comparison rows")
    return rows


def run(root, step, device="cuda:0"):
    root = Path(root).resolve()
    if step not in probe.STEPS:
        raise ValueError("Select500/750/1000")
    output = root / f"step{step}"
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((root / "plan.json").read_text())
    source_id = plan.get("source_id", "wood_spoon")
    if source_id not in SOURCE_IDS:
        raise ValueError("Unknown Material source")
    git = git_info()
    if git != plan["git"]:
        raise ValueError("Inference checkout differs from new training")
    locks = json.loads((root / "source_locks.json").read_text())
    index_path = root / "exports" / f"step{step}" / "index.json"
    index = json.loads(index_path.read_text())
    if index["git"] != git or index["step"] != step or {x["method"] for x in index["runs"]} != {"fullrank", "lora_r64"}:
        raise ValueError("Selected step export index differs")
    import torch
    parents = {item["method"]: item for item in json.loads(
        (root / "training/training_index.json").read_text())["runs"]}
    for selection in index["runs"]:
        parent = parents[selection["method"]]
        checkpoint = Path(selection["checkpoint"])
        if (any(item.get("source_id") != source_id or item.get("concept_id") != source_id + "_material"
                or item.get("branch") != "material" for item in (selection, parent))
                or selection["steps"] != step or selection["training_steps"] != 1000
                or selection["source_checkpoint"] != parent["checkpoint"]
                or selection["run"] != parent["run"] or selection["source_hashes"] != parent["hashes"]
                or checkpoint != index_path.parent / "checkpoints" / selection["method"]
                or json.loads((checkpoint / "receipt.json").read_text()) != selection):
            raise ValueError("Derived initializer export provenance differs")
        adapter, embedding, empty, record = step_snapshots._paired(torch, parent, step)
        exported = torch.load(checkpoint / selection["weight"], map_location="cpu")
        token = torch.load(checkpoint / "<M*>.bin", map_location="cpu")
        shapes, _ = step_snapshots.adapter_schema(selection["method"])
        step_snapshots._check_state(torch, exported, shapes, empty, "derived")
        if (record["source_snapshot_hashes"] != selection["source_snapshot_hashes"]
                or set(token) != {"<M*>"}
                or not torch.equal(token["<M*>"], embedding)
                or any(not torch.equal(adapter[key], exported[key]) for key in shapes)):
            raise ValueError("New initializer paired adapter/token export differs from its training snapshot")
        del adapter, embedding, exported, token
        locks.update(selection["hashes"])
        locks[str(checkpoint / "receipt.json")] = probe.schema.sha256(checkpoint / "receipt.json")
    for path in (root / "plan.json", root / "source_locks.json", root / "training/training_index.json",
                 index_path, probe.TRANSFER_PROMPTS):
        locks[str(path)] = probe.schema.sha256(path)
    verify_locks(locks)
    rows = comparison_rows(plan["protocol"], plan["baseline_rows"][str(step)], step, source_id)
    output.mkdir()
    (output / "images").mkdir()
    (output / "generation_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    provenance = {"status": "running", "git": git, "source_id": source_id, "training_steps": step, "initializer": "material",
        "training": index["runs"], "source_locks": locks, "sampling": plan["protocol"]["inference"],
        "baseline_transfer": plan["source_transfer"], "new_images": 16, "reused_controls": 48}
    probe.write(output / "provenance.json", provenance)
    try:
        with (output / "generation_status.jsonl").open("x", encoding="utf8") as ledger:
            for row in rows:
                if row["reused"]:
                    shutil.copyfile(row["source_image"], output / row["image_path"])
                    if probe.schema.sha256(output / row["image_path"]) != row["image_sha256"]:
                        raise ValueError("Copied old control image differs")
                    ledger.write(json.dumps(row) + "\n")
        for selection in index["runs"]:
            pipe = probe.load_pipeline([selection], device)
            chosen = [row for row in rows if not row["reused"] and row["method"] == selection["method"]]
            probe.generate(pipe, chosen, output, plan["protocol"]["inference"], device, True)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
        generated = [json.loads(line) for line in (output / "generation_status.jsonl").read_text().splitlines()]
        expected = {row["id"]: row for row in rows}
        if len(generated) != 64 or {row["id"] for row in generated} != set(expected):
            raise ValueError("Incomplete initializer transfer ledger")
        for row in generated:
            if (any(row.get(key) != value for key, value in expected[row["id"]].items())
                    or row.get("texture_adapter_loaded") is not False
                    or row.get("material_adapter_loaded") != row["condition"].startswith("learned_")
                    or probe.schema.sha256(output / row["image_path"]) != row["image_sha256"]):
                raise ValueError("Initializer comparison row,adapter or image changed")
        verify_locks(locks)
        if git_info() != git:
            raise ValueError("Checkout changed during transfer")
        sheets = review_bundle(output, generated, index["references"])
        probe.write(output / "summary.json", {"status": "succeeded", "training_steps": step,
            "generated_images": 16, "reused_images": 48, "total_images": 64, "missing_images": 0,
            "contact_sheets": sheets, "safety_filtered": sum(x["status"] == "safety_filtered" for x in generated),
            "human_ratings": "pending", "disentanglement_score": None})
        provenance["status"] = "succeeded"
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        probe.write(output / "provenance.json", provenance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=probe.STEPS, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.run_root, args.step, args.device)
