"""Apply unchanged Experiment B material prompts to audited MARBLE Subject snapshots."""

import argparse
import html
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw

from experiments.sd15_token_local_kv_v1 import evaluate
from experiments.sd15_token_local_kv_v1.contract import read_json, sha256
from experiments.sd_native_comparison_v1.generate import write_json
from experiments.subject_marble_color30_v1 import training
from experiments.subject_multicolor_multimaterial_b.inference import (
    MATERIALS, METHODS, SAMPLING, STEPS, _locks, export_subject_step, prompt_rows,
)
from experiments.texture_material_disentangled_v1 import train as shared
from scripts.launch.colorpeel_run import git_info, read_config

ROOT = training.ROOT


def _selections(index_path, dataset, *, training_module=training):
    index = read_json(index_path)
    runs = index.get("runs", [])
    if (len(runs) != 2 or {run.get("method") for run in runs} != set(METHODS)
            or index.get("git") != git_info()):
        raise ValueError("expected two completed MARBLE Subject methods at the current clean revision")
    result = []
    for selection in runs:
        method, run = selection["method"], Path(selection["run"]).resolve()
        expected = training_module.build_config(method, dataset)
        if (selection.get("status") != "verified" or selection.get("token") != "<S*>"
                or selection.get("steps") != 1000 or selection.get("branch") != "subject"
                or read_config(run / "config.yaml") != expected):
            raise ValueError("Subject selection is unverified or differs from the declared MARBLE recipe")
        _locks(selection.get("hashes", {}))
        recipe = {"method": method, "branch": "subject", "concept_id": "mailbox", "token": "<S*>"}
        audited = shared.audit_run(recipe, run, expected)
        if audited != selection:
            raise ValueError("training index differs from the actual completed Subject audit")
        result.append(audited)
    return result


def _contact_sheet(output, references, rows=(), method=None, *, expected_count=None, label="MARBLE",
                   probe_labels=None, seeds=None, image_root=None):
    """Include every actual training RGB with its full conditioned caption."""
    from experiments.natural_texture_material_v1.probe_review import _font

    if expected_count is None and len(references) not in (14, 35):
        raise ValueError("sheets require all fourteen or thirty-five actual training images")
    if expected_count is not None and len(references) != expected_count:
        raise ValueError(f"sheets require all {expected_count} actual training images")
    font = _font()
    seeds = SAMPLING["seeds"] if seeds is None else seeds
    image_root = output if image_root is None else Path(image_root)
    labels = MATERIALS if probe_labels is None else probe_labels
    columns = max(5, len(labels))
    caption_lines = [textwrap.wrap(item["caption"], 33) for item in references]
    row_height = 235 + 17 * max(len(lines) for lines in caption_lines)
    training_height = 35 + ((len(references) + columns - 1) // columns) * row_height
    sheet = Image.new("RGB", (256 * columns, training_height + (82 + 315 * len(seeds) if rows else 0)), "white")
    draw = ImageDraw.Draw(sheet)
    originals = sum(item["kind"] in ("original_recolor", "original_recolor_cutout") for item in references)
    draw.text((10, 8), f"{label}: {len(references)} ACTUAL training images ({originals} originals + {len(references) - originals} counterfactuals) and captions", font=font, fill="black")
    for number, reference in enumerate(references):
        x, y = (number % columns) * 256 + 5, (number // columns) * row_height + 35
        with Image.open(reference["image"]) as image:
            image = image.copy()
            image.thumbnail((200, 200))
            sheet.paste(image, (x, y))
        draw.text((x, y + 203), str(reference["id"]), font=font, fill="black")
        for line, text in enumerate(caption_lines[number]):
            draw.text((x, y + 222 + line * 17), text, font=font, fill="black")
    if rows:
        draw.text((10, training_height + 8), f"{method}; checkpoint {rows[0]['step']}; PNDM100 / CFG3.5 / 512 / fp16 / safety checker", font=font, fill="black")
        by_key = {(row.get("condition", row.get("material")), row["seed"]): row for row in rows}
        for column, condition in enumerate(labels):
            draw.text((column * 256 + 5, training_height + 34), condition, font=font, fill="black")
            for line, seed in enumerate(seeds):
                row = by_key[condition, seed]
                x, y = column * 256 + 5, training_height + 60 + line * 315
                with Image.open(image_root / row["image_path"]) as image:
                    sheet.paste(image.convert("RGB").resize((246, 246)), (x, y))
                draw.text((x, y + 250), f"seed {seed}: {row['status']}", font=font,
                          fill="red" if row["status"] == "safety_filtered" else "black")
    name = f"{method}_contact_sheet.jpg" if method else "training_contact_sheet.jpg"
    sheet.save(output / name, quality=92)
    return name


def run(dataset, output, step, training_index=None, execute_inference=False, device="cuda:0", *,
        training_module=training, reference_count=None, experiment="subject_marble_color30_v1",
        reference_label="MARBLE", probe_rows=None, probe_labels=None,
        selection_loader=None, baseline_note=None, sampling_seeds=None, image_root=None):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if (output.exists() or output.is_relative_to(training_module.ROOT) or dataset.is_relative_to(output)
            or output.is_relative_to(dataset)):
        raise ValueError("inference output must be new and outside the repository and dataset")
    if execute_inference and training_index is None:
        raise ValueError("explicit inference requires a completed training index")
    training_module.verify_dataset(dataset)
    rows = prompt_rows(step) if probe_rows is None else probe_rows(step)
    expected_images = len(rows)
    seeds = SAMPLING["seeds"] if sampling_seeds is None else tuple(sampling_seeds)
    if (sampling_seeds is not None and (not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or {row["seed"] for row in rows} != set(seeds))):
        raise ValueError("explicit sampling seeds must uniquely match the probe rows")
    sampling = SAMPLING if sampling_seeds is None else {**SAMPLING, "seeds": list(seeds)}
    explicit_image_root = image_root is not None
    image_root = output if image_root is None else Path(image_root).resolve()
    targets = {}
    if explicit_image_root:
        if (not image_root.is_dir() or image_root.is_relative_to(training_module.ROOT)
                or image_root.is_relative_to(ROOT) or image_root.is_relative_to(dataset)
                or dataset.is_relative_to(image_root)):
            raise ValueError("explicit image root must be an existing external directory outside code and dataset")
        for row in rows:
            relative = Path(row["image_path"])
            target = (image_root / relative).resolve()
            if (relative.is_absolute() or relative.drive or ".." in relative.parts
                    or not target.is_relative_to(image_root) or target == image_root):
                raise ValueError("probe image path must stay relative and inside image root")
            if target.exists():
                raise FileExistsError(target)
            targets[row["id"]] = target
        if len(targets) != len(rows) or len(set(targets.values())) != len(rows):
            raise ValueError("probe image paths and IDs must be unique")
    references = [json.loads(line) for line in (dataset / "training_assets_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    locks = {str(dataset / name): sha256(dataset / name) for name in
             ("concepts.json", "training_assets_manifest.jsonl", "source_locks.json", "dataset_audit.json")}
    locks.update({item[field]: item[field + "_sha256"] for item in references for field in ("image", "mask")})
    if training_index is not None:
        training_index = Path(training_index).resolve()
        locks[str(training_index)] = sha256(training_index)
    if execute_inference:
        selections = (_selections(training_index, dataset, training_module=training_module)
                      if selection_loader is None else selection_loader(training_index, dataset))
    else:
        selections = []
    if any(output.is_relative_to(Path(selection["run"]).resolve())
           or Path(selection["run"]).resolve().is_relative_to(output) for selection in selections):
        raise ValueError("inference output must be outside training runs")
    if explicit_image_root and any(image_root.is_relative_to(Path(selection["run"]).resolve())
            or Path(selection["run"]).resolve().is_relative_to(image_root) for selection in selections):
        raise ValueError("image root must be outside training runs")
    _locks(locks)
    output.mkdir(parents=True)
    provenance = {"status": "running" if execute_inference else "dry_run", "step": step,
                  "experiment": experiment, "git": git_info(), "dataset": str(dataset),
                  "source_hashes": locks, "training_index": str(training_index) if training_index else None,
                  "sampling": sampling, "expected_images": expected_images, "generated_images": 0,
                  "training_images": len(references), "training": selections,
                  "source_mode": read_json(dataset / "dataset_audit.json").get("source_mode", "original_scene"),
                  "selected_colors": read_json(dataset / "dataset_audit.json").get("selected_colors"),
                  "baseline": baseline_note if baseline_note is not None else (
                               "Experiment B prompts and sampling reused; no baseline images copied or regenerated."
                               if probe_rows is None else
                               "Original Experiment B material prompts and sampling reused, plus four declared independent context/color probes; no baseline images copied or regenerated.")}
    if explicit_image_root:
        provenance["image_root"] = str(image_root)
    write_json(output / "provenance.json", provenance)
    write_json(output / "training_references.json", {"images": references})
    (output / "generation_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    sheets = [_contact_sheet(output, references, expected_count=reference_count, label=reference_label,
                             probe_labels=probe_labels, seeds=seeds, image_root=image_root)]
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
                    image_path = targets[row["id"]] if explicit_image_root else output / row["image_path"]
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    if explicit_image_root:
                        with image_path.open("xb") as handle:
                            result.images[0].save(handle, format="PNG")
                    else:
                        result.images[0].save(image_path)
                    generated.append({**row, "status": "safety_filtered" if result.nsfw_content_detected[0] else "ok",
                                      "image_sha256": sha256(image_path)})
                    with (output / "generation_status.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(generated[-1]) + "\n")
                _locks(derived["hashes"])
                del pipe
                torch.cuda.empty_cache()
                sheets.append(_contact_sheet(output, references, [row for row in generated if row["method"] == selection["method"]], selection["method"], expected_count=reference_count, label=reference_label, probe_labels=probe_labels, seeds=seeds, image_root=image_root))
            if len(generated) != expected_images or len({row["id"] for row in generated}) != expected_images:
                raise ValueError("incomplete Subject inference matrix")
            provenance.update(status="succeeded", generated_images=expected_images,
                              safety_filtered=sum(row["status"] == "safety_filtered" for row in generated))
        _locks(locks)
        training_module.verify_dataset(dataset)
    except BaseException as error:
        provenance.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        provenance["contact_sheets"] = sheets
        write_json(output / "provenance.json", provenance)
        (output / "index.html").write_text(f'<!doctype html><meta charset="utf-8"><title>{html.escape(reference_label)} Subject material prompts</title>'
            + f'<p>Status: {html.escape(provenance["status"])}; checkpoint {step}; {len(references)} actual training images.</p>'
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
