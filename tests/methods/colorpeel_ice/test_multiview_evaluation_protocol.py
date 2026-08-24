import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

from PIL import Image
import pytest
import yaml

from src.methods.colorpeel_ice import multiview_evaluation_protocol as protocol_lib
from src.methods.colorpeel_ice import prepare_multiview_evaluation as prepare
from src.methods.colorpeel_ice import bundle_multiview_evaluation as bundle_lib


ROOT = Path(__file__).parents[3]
PROTOCOL_PATH = (
    ROOT
    / "experiments"
    / "clevr_subject_color_3x3"
    / "manifests"
    / "clevr_multiview_heldout_eval_v1.json"
)
GENERATOR_PATH = (
    ROOT
    / "scripts"
    / "methods"
    / "colorpeel_ice"
    / "generate_multiview_heldout.py"
)
QWEN_PATH = ROOT / "scripts" / "methods" / "colorpeel_ice" / "predict_qwen.py"
SCORER_PATH = (
    ROOT / "scripts" / "methods" / "colorpeel_ice" / "score_multiview_heldout.py"
)


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_multiview_heldout", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_campaign_manifest():
    protocol = protocol_lib.read_evaluation_protocol(PROTOCOL_PATH)
    return [
        {
            **row,
            "model_fingerprint_sha256": f"fingerprint-{fold_id}-{training_seed}",
            "parent_training_variant": (
                f"multiview_v2_fold_{fold_id.lower()}_seed{training_seed}"
            ),
            "dtype": "float16",
            "safety_checker_disabled": True,
            "safety_risk_acknowledged": True,
            "protocol_fingerprint_sha256": protocol["_source_sha256"],
        }
        for fold_id in ("A", "B", "C")
        for training_seed in (42, 43, 44)
        for row in protocol_lib.build_manifest(
            protocol, fold_id=fold_id, training_seed=training_seed
        )
    ]


def test_complete_bundle_manifest_has_locked_seen_and_heldout_counts():
    protocol = protocol_lib.read_evaluation_protocol(PROTOCOL_PATH)
    all_rows = []
    for fold_id in ("A", "B", "C"):
        rows = protocol_lib.build_manifest(
            protocol, fold_id=fold_id, training_seed=42
        )
        assert len(rows) == 180
        assert len({row["id"] for row in rows}) == 180
        assert len({row["image_path"] for row in rows}) == 180
        assert Counter(row["combination_status"] for row in rows) == {
            "seen": 120,
            "held_out": 60,
        }
        assert {row["generation_seed"] for row in rows} == set(range(42, 62))
        assert {row["category"] for row in rows} == {"multiview_grid"}
        assert all(row["prompt"].count("<s") == 1 for row in rows)
        assert all(row["prompt"].count("<c") == 1 for row in rows)
        assert all(row["num_inference_steps"] == 100 for row in rows)
        assert all(row["guidance_scale"] == 6.0 for row in rows)
        all_rows.extend(rows)

    held_out_counts = Counter(
        (row["subject_label"], row["color_label"])
        for row in all_rows
        if row["combination_status"] == "held_out"
    )
    assert set(held_out_counts.values()) == {20}
    assert len(held_out_counts) == 9


def test_generator_dry_run_writes_180_rows_and_explicit_safety_flags(tmp_path):
    generator = load_generator()
    generator.main(
        [
            "--output-dir",
            str(tmp_path),
            "--evaluation-protocol",
            str(PROTOCOL_PATH),
            "--fold-id",
            "B",
            "--training-seed",
            "43",
            "--disable-safety-checker",
            "--acknowledge-safety-risk",
            "--dry-run",
        ]
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "generation_manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(rows) == 180
    assert all(row["fold_id"] == "B" for row in rows)
    assert all(row["training_seed"] == 43 for row in rows)
    assert all(row["safety_checker_disabled"] for row in rows)
    assert all(row["safety_risk_acknowledged"] for row in rows)


def test_generation_status_records_valid_and_missing_images(tmp_path):
    generator = load_generator()
    rows = [
        {"id": "valid", "image_path": "images/valid.png"},
        {"id": "missing", "image_path": "images/missing.png"},
    ]
    valid = tmp_path / "images" / "valid.png"
    valid.parent.mkdir()
    Image.new("RGB", (512, 512), (10, 20, 30)).save(valid)
    statuses = generator.output_status(rows, tmp_path, None)
    assert statuses[0]["status"] == "ok"
    assert len(statuses[0]["image_sha256"]) == 64
    assert statuses[1]["status"] == "failure"
    assert statuses[1]["failure_reason"] == "missing_or_invalid_image"


def test_resume_requires_matching_status_hash_and_fingerprints(tmp_path):
    generator = load_generator()
    row = {
        "id": "item",
        "image_path": "images/item.png",
        "model_fingerprint_sha256": "model",
        "protocol_fingerprint_sha256": "protocol",
    }
    image_path = tmp_path / row["image_path"]
    image_path.parent.mkdir()
    Image.new("RGB", (512, 512), (1, 2, 3)).save(image_path)
    args = argparse.Namespace(
        skip_existing=True,
        output_dir=tmp_path,
        status_path=tmp_path / "generation_status.jsonl",
    )
    with pytest.raises(RuntimeError, match="no generation status ledger"):
        generator.resume_pending_rows([row], args)

    status = generator.output_status([row], tmp_path, None)[0]
    generator.write_jsonl([status], args.status_path)
    assert generator.resume_pending_rows([row], args) == []

    status["model_fingerprint_sha256"] = "wrong"
    generator.write_jsonl([status], args.status_path)
    with pytest.raises(RuntimeError, match="model fingerprint mismatch"):
        generator.resume_pending_rows([row], args)


def write_fake_training_run(root: Path, fold_id: str, seed: int, commit: str) -> None:
    variant = prepare.expected_variant(fold_id, seed)
    run_dir = root / f"run-{fold_id.lower()}-{seed}"
    checkpoint_dir = run_dir / "checkpoints"
    (checkpoint_dir / "checkpoint-1000").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    manifest = {
        "status": "succeeded",
        "returncode": 0,
        "run": {"variant": variant, "seed": seed},
        "git": {"commit": commit, "branch": "test"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = {
        "args": {
            "max_train_steps": 1500,
            "cos_weight": 0.2,
            "adam_weight_decay": 0.01,
            "initializer_token": "cube+sphere+cylinder+red+turquoise+gray",
            "mixed_precision": "no",
        },
        "protocol": {"fold_id": fold_id, "gt_masks_in_training": False},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    metrics = []
    for step in range(1, 1501):
        metrics.append(
            json.dumps(
                {
                    "step": step,
                    "reconstruction_loss": 1.0,
                    "caa_loss": 0.1,
                    "caa_weighted_loss": 0.02,
                    "total_loss": 1.02,
                    "learning_rate": 1e-5,
                }
            )
        )
    (checkpoint_dir / "training_metrics.jsonl").write_text(
        "\n".join(metrics) + "\n", encoding="utf-8"
    )
    tokens = ["<s1*>", "<s2*>", "<s3*>", "<c1*>", "<c2*>", "<c3*>"]
    audit = {
        "observed_optimization_steps": 1500,
        "modifier_tokens": [
            {
                "token": token,
                "exposure_steps": 500,
                "nonzero_gradient_steps": 500,
                "initial_final_l2_delta": 0.1,
            }
            for token in tokens
        ],
        "non_modifier_embedding_drift": {"enforced": False},
    }
    (checkpoint_dir / "embedding_update_audit.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    for name in prepare.FINAL_ARTIFACTS:
        (checkpoint_dir / name).write_bytes(b"weights")
    (run_dir / "logs" / "stdout.log").write_text(
        "\n".join(f"Loaded textual inversion embedding for {token}." for token in tokens),
        encoding="utf-8",
    )


def test_planner_validates_nine_runs_and_derives_nine_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prepare,
        "FINAL_ARTIFACTS",
        tuple(f"artifact-{index}.bin" for index in range(7)),
    )
    training_root = tmp_path / "training"
    training_root.mkdir()
    commit = "a" * 40
    for fold_id in ("A", "B", "C"):
        for seed in (42, 43, 44):
            write_fake_training_run(training_root, fold_id, seed, commit)
    output_dir = tmp_path / "plan"
    result = prepare.plan(
        argparse.Namespace(
            training_root=training_root,
            training_commit=commit,
            evaluation_protocol=PROTOCOL_PATH,
            output_dir=output_dir,
        )
    )
    assert result["status"] == "planned"
    assert result["training_runs"] == 9
    assert result["expected_images"] == 1620
    configs = [json.loads(Path(path).read_text()) for path in result["generation_configs"]]
    assert len(configs) == 9
    assert {config["stage"] for config in configs} == {"generate_multiview"}
    assert all(config["protocol"]["expected_images"] == 180 for config in configs)
    assert all(config["args"]["disable-safety-checker"] is True for config in configs)
    bundle_config = json.loads(Path(result["bundle_config"]).read_text(encoding="utf-8"))
    assert bundle_config["stage"] == "bundle_multiview"
    assert bundle_config["protocol"]["expected_images"] == 1620


def test_qwen_accepts_only_complete_1620_row_multiview_campaign():
    qwen = load_script("predict_qwen_multiview", QWEN_PATH)
    protocol = protocol_lib.read_evaluation_protocol(PROTOCOL_PATH)
    rows = build_campaign_manifest()
    assert len(qwen.multiview_items(rows, protocol)) == 1620
    try:
        qwen.multiview_items(rows[:-1], protocol)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete multiview campaign was accepted")


def test_multiview_scorer_reports_seen_heldout_cells_and_interventions():
    scorer = load_script("score_multiview_heldout", SCORER_PATH)
    protocol = protocol_lib.read_evaluation_protocol(PROTOCOL_PATH)
    manifest = build_campaign_manifest()
    predictions = [
        {
            "id": row["id"],
            "status": "ok",
            "predicted_shape": row["expected_shape"],
            "predicted_color": row["expected_color"],
            "failure_reason": None,
        }
        for row in manifest
    ]
    metrics, scored, tables = scorer.score(manifest, predictions, protocol)
    assert len(scored) == 1620
    assert metrics["overall"]["joint_accuracy_all_expected"] == 1.0
    assert metrics["by_split"]["seen"]["expected"] == 1080
    assert metrics["by_split"]["held_out"]["expected"] == 540
    assert len(tables["checkpoint_split"]) == 18
    assert len(tables["cell"]) == 81
    assert len(tables["fixed_subject"]) == 540
    assert len(tables["fixed_color"]) == 540
    assert all(row["all_three_joint_correct"] for row in tables["fixed_subject"])
    assert all(row["all_three_joint_correct"] for row in tables["fixed_color"])

    predictions[0] = {
        "id": manifest[0]["id"],
        "status": "failure",
        "predicted_shape": None,
        "predicted_color": None,
        "failure_reason": "model_error",
    }
    metrics, _, tables = scorer.score(manifest, predictions, protocol)
    assert metrics["overall"]["prediction_failures"] == 1
    assert metrics["overall"]["joint_accuracy_all_expected"] == 1619 / 1620
    assert metrics["overall"]["joint_accuracy_valid_only"] == 1.0
    assert len(tables["failures"]) == 1


def test_campaign_validator_rejects_wrong_fold_mapping_and_sampling():
    protocol = protocol_lib.read_evaluation_protocol(PROTOCOL_PATH)
    rows = build_campaign_manifest()
    assert len(protocol_lib.validate_campaign_manifest(rows, protocol)) == 1620
    rows[0]["combination_status"] = "seen"
    rows[0]["held_out"] = False
    rows[20]["combination_status"] = "held_out"
    rows[20]["held_out"] = True
    with pytest.raises(ValueError, match="fold mapping"):
        protocol_lib.validate_campaign_manifest(rows, protocol)
    rows = build_campaign_manifest()
    rows[0]["dtype"] = "float32"
    with pytest.raises(ValueError, match="dtype"):
        protocol_lib.validate_campaign_manifest(rows, protocol)


def test_planner_rejects_duplicate_modifier_token(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prepare,
        "FINAL_ARTIFACTS",
        tuple(f"artifact-{index}.bin" for index in range(7)),
    )
    training_root = tmp_path / "training"
    training_root.mkdir()
    commit = "b" * 40
    write_fake_training_run(training_root, "A", 42, commit)
    run_dir = training_root / "run-a-42"
    audit_path = run_dir / "checkpoints" / "embedding_update_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["modifier_tokens"][-1]["token"] = "<c2*>"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="modifier-token audit failed"):
        prepare.validate_training_run(run_dir, commit)


def test_qwen_resume_accepts_only_unique_known_completed_ids(tmp_path):
    qwen = load_script("predict_qwen_resume", QWEN_PATH)
    items = build_campaign_manifest()
    output = tmp_path / "predictions.jsonl"
    output.write_text(
        json.dumps({"id": items[0]["id"], "status": "ok"}) + "\n",
        encoding="utf-8",
    )
    assert qwen.completed_ids(output, items, True) == {items[0]["id"]}
    with pytest.raises(FileExistsError):
        qwen.completed_ids(output, items, False)
    output.write_text(
        "\n".join(
            json.dumps({"id": items[0]["id"], "status": "ok"})
            for _ in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        qwen.completed_ids(output, items, True)


def test_bundle_merges_nine_generation_runs_with_absolute_images(tmp_path):
    protocol = protocol_lib.read_evaluation_protocol(PROTOCOL_PATH)
    generation_root = tmp_path / "runs"
    generation_root.mkdir()
    image_index = 0
    for fold_id in ("A", "B", "C"):
        for training_seed in (42, 43, 44):
            variant = bundle_lib.expected_variant(fold_id, training_seed)
            run_dir = generation_root / f"run-{fold_id.lower()}-{training_seed}"
            inference_dir = run_dir / "inference"
            inference_dir.mkdir(parents=True)
            launcher_manifest = {
                "status": "succeeded",
                "returncode": 0,
                "stage": "generate_multiview",
                "run": {"variant": variant, "seed": training_seed},
                "git": {"commit": "c" * 40, "branch": "test"},
            }
            (run_dir / "manifest.json").write_text(
                json.dumps(launcher_manifest), encoding="utf-8"
            )
            rows = []
            statuses = []
            for row in protocol_lib.build_manifest(
                protocol, fold_id=fold_id, training_seed=training_seed
            ):
                image_index += 1
                row = {
                    **row,
                    "model_fingerprint_sha256": f"model-{fold_id}-{training_seed}",
                    "protocol_fingerprint_sha256": protocol["_source_sha256"],
                    "parent_training_variant": (
                        f"multiview_v2_fold_{fold_id.lower()}_seed{training_seed}"
                    ),
                    "dtype": "float16",
                    "safety_checker_disabled": True,
                    "safety_risk_acknowledged": True,
                }
                image_path = inference_dir / row["image_path"]
                image_path.parent.mkdir(parents=True, exist_ok=True)
                color = (
                    image_index % 256,
                    (image_index // 256) % 256,
                    (image_index // (256 * 256)) % 256,
                )
                Image.new("RGB", (512, 512), color).save(image_path)
                rows.append(row)
                statuses.append(
                    {
                        "id": row["id"],
                        "status": "ok",
                        "image_sha256": bundle_lib.sha256_file(image_path),
                        "model_fingerprint_sha256": row["model_fingerprint_sha256"],
                        "protocol_fingerprint_sha256": row["protocol_fingerprint_sha256"],
                    }
                )
            bundle_lib.write_jsonl(rows, inference_dir / "generation_manifest.jsonl")
            bundle_lib.write_jsonl(statuses, inference_dir / "generation_status.jsonl")

    output_dir = tmp_path / "bundle"
    result = bundle_lib.bundle(
        argparse.Namespace(
            generation_root=generation_root,
            evaluation_protocol=PROTOCOL_PATH,
            output_dir=output_dir,
        )
    )
    assert result["status"] == "ready_for_human_review"
    merged = bundle_lib.read_jsonl(output_dir / "campaign_generation_manifest.jsonl")
    assert len(merged) == 1620
    assert all(Path(row["image_path"]).is_absolute() for row in merged)
    assert len((output_dir / "human_review.csv").read_text(encoding="utf-8").splitlines()) == 1621
    assert len(result["contact_sheets"]) == 9
    assert all(Path(path).is_file() for path in result["contact_sheets"])
    qwen_config = json.loads((output_dir / "qwen_config.json").read_text(encoding="utf-8"))
    assert qwen_config["args"]["protocol"] == "multiview-heldout"
