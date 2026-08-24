#!/usr/bin/env python3
"""Create an auditable ColorPeel run directory and launch one tracked stage."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # JSON-formatted YAML remains supported without PyYAML.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGES = {
    "prepare": "src/methods/colorpeel_ice/prepare_clevr_3x3.py",
    "train": "src/train/train_colorpeel.py",
    "generate": "scripts/methods/colorpeel_ice/generate.py",
    "generate_multiview": "scripts/methods/colorpeel_ice/generate_multiview_heldout.py",
    "bundle_multiview": "src/methods/colorpeel_ice/bundle_multiview_evaluation.py",
    "segment": "scripts/methods/colorpeel_ice/segment_grounded_sam.py",
    "predict_qwen": "scripts/methods/colorpeel_ice/predict_qwen.py",
    "score_clevr": "scripts/methods/colorpeel_ice/score_clevr_predictions.py",
    "score_color": "scripts/methods/colorpeel_ice/evaluate_color_metrics.py",
    "score_multiview": "scripts/methods/colorpeel_ice/score_multiview_heldout.py",
}
RESUMABLE_STAGES = {"generate_multiview", "predict_qwen"}
MANAGED_ARGUMENTS = {
    "output_dir",
    "output-dir",
    "output",
    "merged-output",
}
RUN_ID_PATTERN = re.compile(
    r"^\d{8}-\d{6}__[A-Za-z0-9][A-Za-z0-9_-]*__"
    r"[A-Za-z0-9][A-Za-z0-9_-]*__[0-9a-f]{7,40}__[-+]?\d+$"
)
ENV_PATTERN = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def read_config(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if yaml is None:
        config = json.loads(raw)
    else:
        config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("config must be a mapping")
    missing = {"stage", "run", "args"} - set(config)
    if missing:
        raise ValueError("missing config fields: " + ", ".join(sorted(missing)))
    if config["stage"] not in STAGES:
        raise ValueError(f"unknown stage: {config['stage']}")
    if not isinstance(config["run"], dict) or not isinstance(config["args"], dict):
        raise ValueError("run and args must be mappings")
    for key in ("study", "variant", "seed"):
        if key not in config["run"]:
            raise ValueError(f"missing run.{key}")
    stage_managed_arguments = set(MANAGED_ARGUMENTS)
    if config["stage"] == "segment":
        stage_managed_arguments.add("mask-dir")
    managed = stage_managed_arguments & set(config["args"])
    if managed:
        raise ValueError("output arguments are launcher-managed: " + ", ".join(sorted(managed)))
    return config


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def git_info() -> dict[str, str]:
    if git_output("status", "--porcelain"):
        raise RuntimeError("refusing to launch from a dirty worktree")
    return {
        "commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
    }


def resolve_run_dir(run_dir: Path) -> Path:
    value = os.environ.get("COLORPEEL_RUN_ROOT")
    if not value:
        raise EnvironmentError("COLORPEEL_RUN_ROOT must be set")
    root = Path(os.path.expandvars(value)).expanduser().resolve()
    resolved = Path(os.path.expandvars(str(run_dir))).expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"--run-dir must be below COLORPEEL_RUN_ROOT: {root}") from error
    if resolved == root:
        raise ValueError("--run-dir must name one run below COLORPEEL_RUN_ROOT")
    return resolved


def validate_run_id(run_dir: Path, config: dict[str, Any], commit: str) -> None:
    run = config["run"]
    suffix = f"__{run['study']}__{run['variant']}__{commit[:7]}__{run['seed']}"
    if not RUN_ID_PATTERN.fullmatch(run_dir.name) or not run_dir.name.endswith(suffix):
        raise ValueError(
            "run directory must be TIMESTAMP__study__variant__commit7__seed; "
            f"expected suffix {suffix}"
        )


def expand_value(value: Any, environment: dict[str, str]) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group("braced") or match.group("plain")
            if name not in environment:
                raise EnvironmentError(f"missing environment variable: {name}")
            return environment[name]

        return ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [expand_value(item, environment) for item in value]
    return value


def argument_tokens(arguments: dict[str, Any], environment: dict[str, str]) -> list[str]:
    tokens: list[str] = []
    for key, raw_value in arguments.items():
        value = expand_value(raw_value, environment)
        if value is None or value is False:
            continue
        tokens.append("--" + key)
        if value is True:
            continue
        if isinstance(value, list):
            tokens.extend(str(item) for item in value)
        elif isinstance(value, (str, int, float)):
            tokens.append(str(value))
        else:
            raise ValueError(f"unsupported argument value for {key}")
    return tokens


def managed_output_args(stage: str, run_dir: Path) -> dict[str, str]:
    if stage == "prepare":
        return {"output-dir": str(run_dir / "data")}
    if stage == "train":
        return {"output_dir": str(run_dir / "checkpoints")}
    if stage in {"generate", "generate_multiview"}:
        return {"output-dir": str(run_dir / "inference")}
    if stage == "bundle_multiview":
        return {"output-dir": str(run_dir / "evaluation" / "campaign")}
    if stage == "segment":
        return {
            "mask-dir": str(run_dir / "evaluation" / "masks"),
            "output": str(run_dir / "evaluation" / "segmentation_status.jsonl"),
        }
    if stage == "predict_qwen":
        return {"output": str(run_dir / "evaluation" / "qwen_predictions.jsonl")}
    if stage == "score_clevr":
        return {
            "output": str(run_dir / "evaluation" / "clevr_metrics.json"),
            "merged-output": str(run_dir / "evaluation" / "clevr_predictions.jsonl"),
        }
    if stage == "score_color":
        return {"output": str(run_dir / "evaluation" / "color_metrics.csv")}
    if stage == "score_multiview":
        return {"output-dir": str(run_dir / "evaluation" / "multiview_metrics")}
    raise AssertionError(stage)


def build_command(config: dict[str, Any], run_dir: Path, environment: dict[str, str]) -> list[str]:
    stage = config["stage"]
    command = [sys.executable, str(PROJECT_ROOT / STAGES[stage])]
    command.extend(argument_tokens(config["args"], environment))
    command.extend(argument_tokens(managed_output_args(stage, run_dir), environment))
    return command


def write_environment(path: Path, environment: dict[str, str]) -> None:
    lines = [
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
    ]
    for key in sorted(environment):
        if key.startswith("COLORPEEL_") or key == "CUDA_VISIBLE_DEVICES":
            lines.append(f"{key}={environment[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    cli = parse_args(argv)
    config_path = cli.config.resolve()
    config = read_config(config_path)
    revision = git_info()
    run_dir = resolve_run_dir(cli.run_dir)
    validate_run_id(run_dir, config, revision["commit"])

    environment = os.environ.copy()
    for key, value in config.get("environment", {}).items():
        if not isinstance(key, str) or not isinstance(value, (str, int, float)):
            raise ValueError("environment must contain string keys and scalar values")
        environment[key] = str(value)
    environment["COLORPEEL_RUN_DIR"] = str(run_dir)
    command = build_command(config, run_dir, environment)

    manifest_path = run_dir / "manifest.json"
    if cli.resume:
        if config["stage"] not in RESUMABLE_STAGES:
            raise ValueError(f"stage is not resumable: {config['stage']}")
        if not run_dir.is_dir() or not manifest_path.is_file():
            raise FileNotFoundError(f"resume run directory is incomplete: {run_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") not in {"running", "failed"}:
            raise ValueError("only running or failed runs may be resumed")
        if manifest.get("git") != revision:
            raise ValueError("resume requires the exact original Git revision")
        snapshot_config = read_config(run_dir / "config.yaml")
        if snapshot_config != config:
            raise ValueError("resume config does not match the immutable run snapshot")
        if manifest.get("command") != command:
            raise ValueError("resume command does not match the original command")
        if config["stage"] == "generate_multiview":
            if config["args"].get("skip-existing") is not True:
                raise ValueError("multiview resume requires locked skip-existing=true")
        else:
            command = [*command, "--resume"]
        manifest["status"] = "running"
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["resume_started_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest.setdefault("resume_commands", []).append(command)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with (run_dir / "logs" / "stdout.log").open("a", encoding="utf-8") as log:
            log.write(f"\n===== RESUME {manifest['resume_count']} =====\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["returncode"] = result.returncode
        manifest["status"] = "succeeded" if result.returncode == 0 else "failed"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result.returncode

    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    for name in ("logs", "checkpoints", "data", "inference", "evaluation", "figures"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    snapshot = yaml.safe_dump(config, sort_keys=False) if yaml is not None else json.dumps(config, indent=2)
    (run_dir / "config.yaml").write_text(snapshot, encoding="utf-8")
    (run_dir / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\ncd "
        + shlex.quote(str(PROJECT_ROOT))
        + "\n"
        + shlex.join(command)
        + "\n",
        encoding="utf-8",
    )
    write_environment(run_dir / "environment.txt", environment)
    manifest = {
        "status": "dry-run" if cli.dry_run else "running",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": config["stage"],
        "run": config["run"],
        "git": revision,
        "config_source": str(config_path),
        "data_manifest": expand_value(config.get("data_manifest"), environment),
        "command": command,
        "managed_outputs": managed_output_args(config["stage"], run_dir),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if cli.dry_run:
        print(f"Dry run created: {run_dir}")
        return 0

    with (run_dir / "logs" / "stdout.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["returncode"] = result.returncode
    manifest["status"] = "succeeded" if result.returncode == 0 else "failed"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
