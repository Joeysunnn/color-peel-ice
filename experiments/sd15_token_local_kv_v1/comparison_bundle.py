"""Verify completed paired inference and create new SD1.4/SD1.5 review sheets."""

import argparse
import html
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

from experiments.sd15_token_local_kv_v1.contract import read_json, sha256


BUNDLE = "comparison_bundle.json"
INPUTS = ("provenance.json", "generation_manifest.jsonl", "generation_status.jsonl")
NOTE = "Safety-filtered images are missing visual observations, not quality failures."


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def image_path(directory, relative):
    path = (directory / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(directory):
        raise ValueError("generated image path must stay inside its job")
    return path


def verified_pairs(directory):
    if read_json(directory / "provenance.json").get("status") != "succeeded":
        raise ValueError("comparison requires succeeded inference provenance")
    rows, statuses = ([json.loads(line) for line in (directory / name).read_text(
        encoding="utf-8").splitlines()] for name in INPUTS[1:])
    required = {"id", "prompt", "seed", "image_path", "reference_image",
                "reference_sha256", "reference_filtered"}
    if (not rows or any(not required.issubset(row) for row in rows)
            or len({row["id"] for row in rows}) != len(rows)
            or len({row["image_path"] for row in rows}) != len(rows)
            or len(statuses) != len(rows)
            or len({row.get("id") for row in statuses}) != len(rows)):
        raise ValueError("incomplete or duplicate comparison manifest/status rows")
    by_id = {row["id"]: row for row in statuses}
    if set(by_id) != {row["id"] for row in rows}:
        raise ValueError("comparison status IDs differ from its manifest")
    pairs = []
    for row in rows:
        status = by_id[row["id"]]
        if (not isinstance(row["id"], str) or not isinstance(row["prompt"], str)
                or type(row["seed"]) is not int or type(row["reference_filtered"]) is not bool
                or any(status.get(key) != value for key, value in row.items())
                or status.get("status") not in {"ok", "safety_filtered"}):
            raise ValueError("comparison row fields or generation status differ")
        reference = Path(row["reference_image"])
        generated = image_path(directory, row["image_path"])
        if (not reference.is_absolute() or sha256(reference) != row["reference_sha256"]
                or sha256(generated) != status.get("image_sha256")):
            raise ValueError("comparison image hash differs")
        pairs.append({**row, "image_sha256": status["image_sha256"],
                      "generated_filtered": status["status"] == "safety_filtered"})
    return pairs


def font(size):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "Arial.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def wrapped(draw, text, face, width):
    """Wrap by measured pixels, including a word longer than the text column."""
    lines = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for char in paragraph:
            if line and draw.textlength(line + char, font=face) > width:
                lines.append(line.rstrip())
                line = ""
            line += char
        lines.append(line.rstrip())
    return lines


def render_sheet(directory, pairs, path, page):
    from PIL import Image, ImageDraw, ImageOps

    face, heading = font(21), font(27)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    descriptions = []
    for row in pairs:
        steps = ", ".join(f"{key}={value}" for key, value in row.items()
                          if (key.endswith("_step") or key == "num_inference_steps")
                          and value is not None)
        text = f"{row['id']}\nseed={row['seed']}" + (f"; {steps}" if steps else "")
        descriptions.append(wrapped(measure, text + "\n" + row["prompt"], face, 700))
    heights = [max(344, 30 + 27 * len(lines)) for lines in descriptions]
    sheet = Image.new("RGB", (1400, 125 + sum(heights) + 60), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((28, 18), f"SD1.4 / SD1.5 comparison - page {page}", font=heading, fill="black")
    draw.text((28, 70), "SD1.4", font=heading, fill="black")
    draw.text((356, 70), "SD1.5", font=heading, fill="black")
    y = 125
    for row, lines, height in zip(pairs, descriptions, heights):
        for x, source, filtered in (
            (28, Path(row["reference_image"]), row["reference_filtered"]),
            (356, image_path(directory, row["image_path"]), row["generated_filtered"]),
        ):
            with Image.open(source) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (296, 296), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (x + (296 - thumb.width) // 2, y))
            draw.text((x, y + 301), "SAFETY FILTERED - no observation" if filtered else "ok",
                      font=font(16), fill="#b00020" if filtered else "#333333")
        for index, line in enumerate(lines):
            draw.text((684, y + index * 27), line, font=face, fill="black")
        y += height
        draw.line((28, y - 8, 1370, y - 8), fill="#dddddd")
    draw.text((28, y + 14), NOTE, font=font(19), fill="#333333")
    sheet.save(path, quality=90)


def bundle_job(directory):
    directory = Path(directory).resolve()
    pairs = verified_pairs(directory)
    sheets_dir = directory / "comparison_sheets"
    if sheets_dir.exists() or (directory / BUNDLE).exists():
        raise FileExistsError("comparison sheets and bundle must be new")
    sheets_dir.mkdir()
    sheets = []
    for start in range(0, len(pairs), 10):
        page = start // 10 + 1
        path = sheets_dir / f"page_{page:03d}.jpg"
        render_sheet(directory, pairs[start:start + 10], path, page)
        sheets.append({"path": path.relative_to(directory).as_posix(), "sha256": sha256(path)})
    summary = {
        "schema": "sd14_sd15_comparison_bundle/v1", "status": "complete",
        "pair_count": len(pairs), "filtered_counts": {
            "sd14": sum(row["reference_filtered"] for row in pairs),
            "sd15": sum(row["generated_filtered"] for row in pairs)},
        "note": NOTE, "inputs": {name: sha256(directory / name) for name in INPUTS},
        "pairs": pairs, "sheets": sheets,
    }
    write_json(directory / BUNDLE, summary)
    return summary


def verified_bundle(directory):
    summary = read_json(directory / BUNDLE)
    pairs = verified_pairs(directory)
    if (summary.get("schema") != "sd14_sd15_comparison_bundle/v1"
            or summary.get("status") != "complete" or summary.get("pairs") != pairs
            or summary.get("pair_count") != len(pairs)
            or summary.get("filtered_counts") != {
                "sd14": sum(row["reference_filtered"] for row in pairs),
                "sd15": sum(row["generated_filtered"] for row in pairs)}
            or summary.get("inputs") != {name: sha256(directory / name) for name in INPUTS}
            or len(summary.get("sheets", [])) != math.ceil(len(pairs) / 10)):
        raise ValueError("comparison bundle no longer matches its inputs")
    for page, sheet in enumerate(summary["sheets"], 1):
        if (sheet.get("path") != f"comparison_sheets/page_{page:03d}.jpg"
                or sha256(image_path(directory, sheet["path"])) != sheet.get("sha256")):
            raise ValueError("comparison sheet hash or path differs")
    return summary


def bundle_all(outputroot):
    outputroot = Path(outputroot).resolve()
    if (outputroot / "index.html").exists() or (outputroot / "summary.json").exists():
        raise FileExistsError("comparison index and summary must be new")
    jobs = read_json(outputroot / "comparison_plan.json")["jobs"]
    if (not jobs or len({job["id"] for job in jobs}) != len(jobs)
            or any(not Path(job["output"]).is_absolute() for job in jobs)
            or len({str(Path(job["output"]).resolve()) for job in jobs}) != len(jobs)):
        raise ValueError("comparison plan must have unique IDs and absolute job directories")
    # Reject incomplete inference across the entire plan before writing any sheets.
    for job in jobs:
        verified_pairs(Path(job["output"]).resolve())
    results, sections = [], []
    for job in jobs:
        directory = Path(job["output"]).resolve()
        if not (directory / BUNDLE).exists():
            bundle_job(directory)
        summary = verified_bundle(directory)
        results.append({"id": job["id"], "output": str(directory),
                        "bundle_sha256": sha256(directory / BUNDLE), **summary})
        links = []
        for page, sheet in enumerate(summary["sheets"], 1):
            relative = os.path.relpath(directory / sheet["path"], outputroot).replace(os.sep, "/")
            href = html.escape(quote(relative, safe="/."), quote=True)
            links.append(f'<li><a href="{href}">Page {page}</a></li>')
        sections.append(f"<section><h2>{html.escape(str(job['id']))}</h2>"
                        f"<p>{summary['pair_count']} pairs; filtered SD1.4: "
                        f"{summary['filtered_counts']['sd14']}; SD1.5: "
                        f"{summary['filtered_counts']['sd15']}</p><ul>{''.join(links)}</ul></section>")
    result = {"schema": "sd14_sd15_comparison_summary/v1", "status": "complete",
              "job_count": len(results), "pair_count": sum(job["pair_count"] for job in results),
              "filtered_counts": {base: sum(job["filtered_counts"][base] for job in results)
                                  for base in ("sd14", "sd15")}, "note": NOTE, "jobs": results,
              "comparison_plan_sha256": sha256(outputroot / "comparison_plan.json")}
    with (outputroot / "index.html").open("x", encoding="utf-8") as stream:
        stream.write('<!doctype html><html lang="en"><meta charset="utf-8">'
                     '<meta name="viewport" content="width=device-width, initial-scale=1">'
                     '<title>SD1.4 / SD1.5 comparisons</title><body>'
                     f"<h1>SD1.4 / SD1.5 comparisons</h1><p>{NOTE}</p>"
                     + "".join(sections) + "</body></html>")
    result["index_sha256"] = sha256(outputroot / "index.html")
    write_json(outputroot / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output-root", type=Path)
    target.add_argument("--job", type=Path)
    args = parser.parse_args()
    bundle_all(args.output_root) if args.output_root else bundle_job(args.job)
