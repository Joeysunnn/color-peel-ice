"""Prepare Subject material prompts; inference requires explicit execution and audited training."""

import argparse
import html
import itertools
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd15_token_local_kv_v1 import evaluate, snapshots
from experiments.sd_native_comparison_v1.generate import write_json
from experiments.subject_multicolor_multimaterial_b import data, training
from experiments.texture_material_disentangled_v1 import train as shared
from scripts.launch.colorpeel_run import git_info, read_config

METHODS = ("fullrank", "lora_r64")
MATERIALS = ("metallic", "matte", "glossy plastic", "ceramic", "wood")
STEPS = (500, 750, 1000)
SAMPLING = {"scheduler": "PNDM", "num_inference_steps": 100, "guidance_scale": 3.5,
            "width": 512, "height": 512, "dtype": "float16", "safety_checker": True,
            "generator": "fresh_device_generator_per_image", "seeds": [42, 43]}


def prompt_rows(step):
    if type(step) is not int or step not in STEPS:
        raise ValueError("expected a declared Subject checkpoint step")
    return [{"id": f"{method}__{material.replace(' ', '_')}__seed{seed}",
             "method": method, "material": material, "seed": seed, "step": step,
             "prompt": f"a photo of <S*> mailbox with a {material} surface",
             "image_path": f"images/{method}/{material.replace(' ', '_')}__seed{seed}.png"}
            for method, material, seed in itertools.product(METHODS, MATERIALS, (42, 43))]


def _locks(hashes):
    if not hashes or any(not Path(path).is_absolute() or sha256(Path(path)) != value
                         for path, value in hashes.items()):
        raise ValueError("Subject source artifact changed or source locks are missing")


def _selections(index_path, dataset):
    index = read_json(index_path)
    runs = index.get("runs", [])
    if (len(runs) != 2 or {run.get("method") for run in runs} != set(METHODS)
            or index.get("git") != git_info()):
        raise ValueError("expected two completed Subject methods at the current clean revision")
    result = []
    for selection in runs:
        method, run = selection["method"], Path(selection["run"]).resolve()
        expected = training.build_config(method, dataset)
        if (selection.get("status") != "verified" or selection.get("token") != "<S*>"
                or selection.get("steps") != 1000 or selection.get("branch") != "subject"
                or read_config(run / "config.yaml") != expected):
            raise ValueError("Subject selection is unverified or differs from Experiment B")
        _locks(selection.get("hashes", {}))
        recipe = {"method": method, "branch": "subject", "concept_id": "mailbox", "token": "<S*>"}
        audited = shared.audit_run(recipe, run, expected)
        if audited != selection:
            raise ValueError("training index differs from the actual completed Subject audit")
        result.append(audited)
    return result


def export_subject_step(selection, step, output, *, steps=STEPS, total_steps=1000):
    """Export the adapter and S embedding from the same audited Accelerator snapshot."""
    import torch

    if (selection.get("token") != "<S*>" or selection.get("branch") != "subject"
            or selection.get("status") != "verified" or selection.get("method") not in METHODS):
        raise ValueError("Subject-only paired export requires an audited S selection")
    _locks(selection["hashes"])
    config = read_config(Path(selection["run"]) / "config.yaml")
    if (step not in steps or step not in snapshots.checkpoint_plan(config["args"])
            or selection["steps"] != total_steps or config["args"]["max_train_steps"] != total_steps):
        raise ValueError("requested Subject snapshot is outside the audited plan")
    output = snapshots._new_output(output, Path(selection["run"]).resolve())
    checkpoint = Path(selection["checkpoint"])
    directory = checkpoint / f"checkpoint-{step}"
    if any(not (directory / name).is_file() or not (directory / name).stat().st_size
           for name in snapshots.STATE_FILES):
        raise ValueError("Subject snapshot has missing or empty training state")
    token_id, _ = snapshots._embedding_audit(checkpoint / "embedding_update_audit.json", total_steps, "<S*>")
    shapes, empty = snapshots.adapter_schema(selection["method"])
    adapter_path = directory / "pytorch_model.bin"
    encoder_path = directory / "pytorch_model_1.bin"
    source_hashes = {str(path.resolve()): sha256(path) for path in (adapter_path, encoder_path)}
    adapter = torch.load(adapter_path, map_location="cpu")
    snapshots._check_state(torch, adapter, shapes, {}, f"Subject step {step}")
    encoder = torch.load(encoder_path, map_location="cpu")
    table = encoder.get(snapshots.EMBEDDING_KEY) if isinstance(encoder, dict) else None
    if (not isinstance(table, torch.Tensor) or table.ndim != 2 or table.shape[1] != 768
            or not 0 <= token_id < table.shape[0]):
        raise ValueError("Subject snapshot embedding table or token ID differs")
    embedding = table[token_id].clone().contiguous()
    del encoder, table
    if not bool(torch.isfinite(embedding).all()):
        raise ValueError("Subject paired embedding is nonfinite")
    if step == total_steps:
        final = torch.load(checkpoint / selection["weight"], map_location="cpu")
        snapshots._check_state(torch, final, shapes, empty, "final Subject")
        final_token = torch.load(checkpoint / "<S*>.bin", map_location="cpu")
        if (not isinstance(final_token, dict) or set(final_token) != {"<S*>"}
                or not torch.equal(final_token["<S*>"], embedding)
                or any(not torch.equal(adapter[key], final[key]) for key in shapes)):
            raise ValueError("final Subject snapshot differs from its paired export")
    _locks({**selection["hashes"], **source_hashes})
    record = {"source_adapter_sha256": source_hashes[str(adapter_path.resolve())],
              "source_text_encoder_sha256": source_hashes[str(encoder_path.resolve())]}
    derived = snapshots._write_export(torch, adapter, embedding, empty, selection, step, record, output)
    derived.update(steps=step, training_steps=total_steps, source_snapshot_hashes=source_hashes,
                   paired_embedding_finite=True)
    derived["hashes"] = {**selection["hashes"], **source_hashes, **derived["hashes"]}
    write_json(output / "receipt.json", derived)
    return derived


def _contact_sheet(output, references, rows=(), method=None):
    """Show all actual RGB training images and full captions above the material grid."""
    from experiments.natural_texture_material_v1.probe_review import _font

    font = _font()
    sheet = Image.new("RGB", (1280, 620 + (700 if rows else 0)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), "Experiment B: ten ACTUAL training images and their material-conditioned captions", font=font, fill="black")
    for number, reference in enumerate(references):
        x, y = (number % 5) * 256 + 5, (number // 5) * 285 + 35
        with Image.open(reference["image"]) as image:
            image = image.copy()
            image.thumbnail((200, 200))
            sheet.paste(image, (x, y))
        for line, text in enumerate(textwrap.wrap(reference["caption"], 33)):
            draw.text((x, y + 203 + line * 17), text, font=font, fill="black")
    if rows:
        draw.text((10, 608), f"{method}; PNDM100 / CFG3.5 / 512 / fp16 / safety checker; columns = material prompts", font=font, fill="black")
        by_key = {(row["material"], row["seed"]): row for row in rows}
        for column, material in enumerate(MATERIALS):
            draw.text((column * 256 + 5, 634), material, font=font, fill="black")
            for line, seed in enumerate((42, 43)):
                row = by_key[material, seed]
                x, y = column * 256 + 5, 660 + line * 315
                with Image.open(output / row["image_path"]) as image:
                    sheet.paste(image.convert("RGB").resize((246, 246)), (x, y))
                draw.text((x, y + 250), f"seed {seed}: {row['status']}", font=font,
                          fill="red" if row["status"] == "safety_filtered" else "black")
    name = f"{method}_contact_sheet.jpg" if method else "training_contact_sheet.jpg"
    sheet.save(output / name, quality=92)
    return name


def run(dataset, output, step, training_index=None, execute_inference=False, device="cuda:0"):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if (output.exists() or output.is_relative_to(data.ROOT) or dataset.is_relative_to(output)
            or output.is_relative_to(dataset)):
        raise ValueError("inference output must be new and outside the repository and dataset")
    if execute_inference and training_index is None:
        raise ValueError("explicit inference requires a completed training index")
    data.verify_dataset(dataset)
    rows = prompt_rows(step)
    references = [json.loads(line) for line in (dataset / "training_assets_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    locks = {str(dataset / name): sha256(dataset / name) for name in
             ("concepts.json", "training_assets_manifest.jsonl", "source_locks.json", "dataset_audit.json")}
    locks.update({item[field]: item[field + "_sha256"] for item in references for field in ("image", "mask")})
    if training_index is not None:
        training_index = Path(training_index).resolve()
        locks[str(training_index)] = sha256(training_index)
    selections = _selections(training_index, dataset) if execute_inference else []
    if any(output.is_relative_to(Path(selection["run"]).resolve())
           or Path(selection["run"]).resolve().is_relative_to(output) for selection in selections):
        raise ValueError("inference output must be outside training runs")
    _locks(locks)
    output.mkdir(parents=True)
    provenance = {"status": "running" if execute_inference else "dry_run", "step": step,
                  "git": git_info(), "dataset": str(dataset), "source_hashes": locks,
                  "training_index": str(training_index) if training_index else None,
                  "sampling": SAMPLING, "expected_images": 20, "generated_images": 0,
                  "baseline": "Not generated: existing baseline does not have every matching step (750).",
                  "training": selections}
    write_json(output / "provenance.json", provenance)
    write_json(output / "training_references.json", {"images": references})
    (output / "generation_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    training_sheet = _contact_sheet(output, references)
    sheets = [training_sheet]
    try:
        if execute_inference:
            import torch
            generated = []
            for selection in selections:
                derived = export_subject_step(selection, step, output / "checkpoints" / selection["method"])
                pipe = evaluate.load_pipeline([derived], device)
                for row in (row for row in rows if row["method"] == selection["method"]):
                    result = pipe(row["prompt"], num_inference_steps=100, guidance_scale=3.5,
                        width=512, height=512, generator=torch.Generator(device=device).manual_seed(row["seed"]),
                        cross_attention_kwargs=evaluate.attention_masks(pipe, row["prompt"], ["<S*>"], 3.5))
                    if result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1:
                        raise ValueError("missing safety checker result")
                    image_path = output / row["image_path"]
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    result.images[0].save(image_path)
                    generated.append({**row, "status": "safety_filtered" if result.nsfw_content_detected[0] else "ok",
                                      "image_sha256": sha256(image_path)})
                    with (output / "generation_status.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(generated[-1]) + "\n")
                _locks(derived["hashes"])
                del pipe
                torch.cuda.empty_cache()
                sheets.append(_contact_sheet(output, references, [row for row in generated if row["method"] == selection["method"]], selection["method"]))
            if len(generated) != 20 or len({row["id"] for row in generated}) != 20:
                raise ValueError("incomplete Subject inference matrix")
            provenance.update(status="succeeded", generated_images=20,
                              safety_filtered=sum(row["status"] == "safety_filtered" for row in generated))
        _locks(locks)
        data.verify_dataset(dataset)
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        provenance["contact_sheets"] = sheets
        write_json(output / "provenance.json", provenance)
        (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Experiment B Subject material prompts</title>'
            + f'<p>Status: {html.escape(provenance["status"])}; checkpoint {step}. No matched baseline is implied.</p>'
            + ''.join(f'<p><a href="{sheet}"><img width="1000" src="{sheet}"></a></p>' for sheet in sheets), encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=int, choices=STEPS, required=True)
    parser.add_argument("--training-index", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-inference", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.execute_inference:
        parser.error("--dry-run and --execute-inference are mutually exclusive")
    run(args.dataset, args.output, args.step, args.training_index, args.execute_inference, args.device)


if __name__ == "__main__":
    main()
