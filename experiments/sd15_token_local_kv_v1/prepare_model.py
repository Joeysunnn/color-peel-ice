"""Create a new read-only-linked SD1.5 bundle with the shared SD1.4 checker."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sd15_token_local_kv_v1.contract import PROTOCOL, read_json, sha256


def main():
    spec = read_json(PROTOCOL)["model"]
    source, checker, output = (Path(spec[key]).resolve()
                               for key in ("source_root", "checker_root", "root"))
    run_root = Path(os.environ["COLORPEEL_RUN_ROOT"]).resolve()
    if not output.is_relative_to(run_root) or output.is_relative_to(ROOT) or output == run_root:
        raise ValueError("new model bundle must be below the external run root")
    if output.exists():
        raise FileExistsError(output)
    downloaded = read_json(source / "download_manifest.json")
    if downloaded["repo"] != spec["repo"] or downloaded["revision"] != spec["revision"]:
        raise ValueError("SD1.5 source revision differs")
    paths = {"model_index.json": source / "model_index.json"}
    for component in ("unet", "vae", "text_encoder", "tokenizer", "scheduler",
                      "safety_checker", "feature_extractor"):
        origin = checker if component in {"safety_checker", "feature_extractor"} else source
        folder = origin / component
        if not folder.is_dir() or not any(folder.iterdir()):
            raise FileNotFoundError(folder)
        paths[component] = folder
    files = {}
    for name, path in paths.items():
        for file in sorted(path.rglob("*")) if path.is_dir() else [path]:
            if file.is_file():
                relative = str(Path(name) / file.relative_to(path)) if path.is_dir() else name
                files[relative] = sha256(file)
    output.mkdir(parents=True)
    for name, path in paths.items():
        (output / name).symlink_to(path, target_is_directory=path.is_dir())
    (output / "model_source.json").write_text(json.dumps(
        {"model": spec, "files": files, "source_download_manifest_sha256": sha256(
            source / "download_manifest.json"), "safety_checker_source": str(checker)},
        indent=2) + "\n", encoding="utf-8")
    print(f"Prepared new SD1.5 bundle; original assets untouched: {output}")


if __name__ == "__main__":
    main()
