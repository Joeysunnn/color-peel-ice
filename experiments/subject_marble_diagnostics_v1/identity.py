"""Identity diagnostics: visible content preservation, never RGB-based albedo verdicts."""

import json

import numpy as np
from PIL import Image


def rows():
    controls = [("none", {}, "no attribute MLP"),
                ("joint_zero", {"metallic": 0, "roughness": 0}, "M0 + R0"),
                *[(f"M{m:+d}", {"metallic": m}, f"M={m:+d} only") for m in (-20, 0, 20)],
                *[(f"R{r:+d}", {"roughness": r}, f"R={r:+d} only") for r in (-1, 0, 1)]]
    return [{"id": f"seed{seed}__{name}", "controls": values, "reference": "self", "target_size": 1024,
             "seed": seed, "control_name": name, "label": f"s{seed} | {label}"}
            for seed in (42, 43, 44) for name, values, label in (controls if seed == 42 else controls[:2])]


def evaluate(original, raw, mask):
    difference = np.abs(np.asarray(raw, dtype=float) - np.asarray(original, dtype=float))
    foreground = np.asarray(mask) > 0
    return {
        "appearance_diagnostics": {"foreground_RGB_MAE_255": float(difference[foreground].mean()),
            "background_RGB_MAE_255": float(difference[~foreground].mean()),
            "interpretation": "RGB appearance differences only; neither albedo leakage nor an identity score."},
        "albedo_preservation": {"status": "not_identifiable_from_RGB_alone",
            "reason": "No intrinsic albedo, geometry or illumination ground truth. No RGB/hue threshold verdict."},
        "content_review": {"texture_patterns": "manual_review_required", "text_or_decorations": "manual_review_required",
            "silhouette_slot_lock_and_parts": "manual_review_required",
            "note": "Reflections may alter image edges. Judge persistent patterns and visible part changes against the input."},
        "flags": ["manual_identity_review_required"], "training_eligible": False}


def zero_offsets(adapter, mlps, target):
    """Measure actual zero-control offsets in CLIP space; does not edit model weights."""
    import torch
    with torch.inference_mode():
        pixels = adapter.clip_image_processor(images=[target], return_tensors="pt").pixel_values
        embedding = adapter.image_encoder(pixels.to(adapter.device, dtype=torch.float16)).image_embeds
        offsets = {name: net(embedding, torch.tensor([0.0], device=adapter.device, dtype=torch.float16))
                   for name, net in mlps.items()}
        joint = sum(offsets.values())
        norm = float(embedding.float().norm())
        return {"reference_embedding_l2": norm,
                "zero_control_offset_l2": {name: float(value.float().norm()) for name, value in offsets.items()},
                "joint_zero_offset_l2": float(joint.float().norm()),
                "note": "CLIP offset magnitudes, not measured material properties. Empty MLP dictionary adds exactly no offset."}


def write_review(output, generated, sheet):
    with Image.open(output / "source/review_mask.png") as image:
        x0, y0, x1, y1 = image.convert("L").getbbox()
    crop = (max(0, x0 - 24), max(0, y0 - 24), min(1024, x1 + 24), min(1024, y1 + 24))
    sources = [("Original mailbox1024", "source/target.png"), ("Same1024 reference", "source/reference_self.png"),
               ("Frozen mailbox Mask", "source/mask.png"), ("Fresh1024 DPT", "source/depth.png")]
    original = {"raw_image": "source/target.png", "label": "Original mailbox input"}
    seed42 = [original] + [row for row in generated if row["seed"] == 42]
    sheet(output, seed42, sources, "seed42_full.png", title="Identity test | original + eight controls | seed42 | NO training")
    sheet(output, seed42, sources, "seed42_subject.png", crop=crop,
          title="Identity test | visible patterns / text / structure | seed42 | Albedo: unverified")
    by_id = {row["id"]: row for row in generated}
    stability = [row for seed in (42, 43, 44) for row in (
        {**original, "label": f"s{seed} | Original input"},
        by_id[f"seed{seed}__none"], by_id[f"seed{seed}__joint_zero"])]
    sheet(output, stability, sources, "stability_subject.png", crop=crop,
          title="Limited seed check | columns: input / no MLP / M0R0 | rows:42,43,44 | NO training")
    neutral = [original] + [by_id["seed42__" + name] for name in ("none", "joint_zero", "M+0", "R+0")]
    # Fixed image-coordinate regions; not inferred geometry or albedo maps.
    regions = {"upper_subject": (x0, y0, x1, y0 + int((y1 - y0) * .55)),
               "lid_and_lock": (x0, y0 + int((y1 - y0) * .32), x1, y1)}
    for name, box in regions.items():
        sheet(output, neutral, sources, f"neutral_{name}.png", crop=box,
              title=f"Identity details: {name} | input / none / joint0 / M0 / R0 | Albedo: unverified")
    (output / "review_regions.json").write_text(json.dumps({"subject_crop": crop, "detail_regions": regions,
        "interpretation": "Fixed visual review crops only. RGB changes are not an albedo or color leakage verdict."}, indent=2) + "\n", encoding="utf8")
