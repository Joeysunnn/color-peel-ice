"""Write independent token-local K/V training configs for the three materials."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.run_root / "configs"
    out.mkdir(exist_ok=False)
    for index, material in enumerate(("mailbox", "metal_spoon", "wood_spoon")):
        staging = args.run_root / "staging" / material
        details = json.loads((staging / "staging_manifest.json").read_text())
        if details["train_images"] != 72:
            raise ValueError(f"Incomplete staging: {material}")
        config = {
            "schema_version": 1,
            "status": "authorized_threeway_material_training",
            "stage": "train",
            "run": {"study": "natural_material_threeway_v1",
                    "variant": f"{material}_token_local_kv_5000", "seed": 42},
            "environment": {"CUDA_VISIBLE_DEVICES": str(index)},
            "data_manifest": str(staging / "training_assets_manifest.jsonl"),
            "source_grid_manifest_sha256": details["source_grid_manifest_sha256"],
            "staging_manifest_sha256": sha(staging / "staging_manifest.json"),
            "args": {
                "pretrained_model_name_or_path": "CompVis/stable-diffusion-v1-4",
                "concepts_list": str(staging / "concepts.json"),
                "resolution": 512, "train_batch_size": 1,
                "learning_rate": 1e-5, "kv_learning_rate": 1e-5,
                "token_local_kv": True, "freeze_model": "crossattn_kv",
                "scale_lr": True, "lr_scheduler": "constant", "lr_warmup_steps": 0,
                "max_train_steps": 5000, "checkpointing_steps": 1000,
                "seed": 42, "mixed_precision": "no", "cos_weight": 0.0,
                "gradient_accumulation_steps": 1, "max_grad_norm": 1.0,
                "dataloader_num_workers": 2,
                "adam_beta1": 0.9, "adam_beta2": 0.999,
                "adam_weight_decay": 0.0, "adam_epsilon": 1e-8,
                "hflip": False, "modifier_token": "<M*>",
                "initializer_token": "material",
            },
        }
        (out / f"{material}.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
