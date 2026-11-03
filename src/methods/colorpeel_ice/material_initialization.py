"""Map visible reflection labels to a bounded initialization prior, not PBR numbers."""

SCHEMA = "colorpeel_vlm_initializers/v2"
POLICY_VERSION = "material_reflection_pool/v1"
INITIALIZERS = ("metallic", "glossy", "matte", "material")
LABELS = {
    "metallicity": ("metallic", "dielectric", "mixed", "uncertain"),
    "roughness": ("low", "medium", "high", "uncertain"),
}
CONFIDENCES = ("high", "medium", "low")


def validate_material(value):
    """Require two separately evidenced classifications; preserve the original value."""
    if not isinstance(value, dict) or set(value) != set(LABELS):
        raise ValueError("Material requires exactly metallicity and roughness")
    for axis, labels in LABELS.items():
        item = value[axis]
        if not isinstance(item, dict) or set(item) != {"label", "confidence", "evidence"}:
            raise ValueError(f"Material {axis} requires label, confidence and evidence")
        if not isinstance(item["label"], str) or item["label"] not in labels:
            raise ValueError(f"Invalid Material {axis} label")
        if not isinstance(item["confidence"], str) or item["confidence"] not in CONFIDENCES:
            raise ValueError(f"Invalid Material {axis} confidence")
        if not isinstance(item["evidence"], str) or not item["evidence"].strip():
            raise ValueError(f"Material {axis} requires nonempty visible evidence")
    return value


def select_material_initializer(value):
    """Select one starting word; the complete two-axis observation remains separate.

    Mixed surfaces use the generic fallback. High-confidence metallic evidence
    otherwise has priority over roughness. Reliable roughness extremes select
    glossy/matte; medium, uncertain or low-confidence roughness uses material.
    These are experiment rules, not calibrated physical or linguistic guarantees.
    """
    validate_material(value)
    metallicity, roughness = value["metallicity"], value["roughness"]
    word, reason, fallback = "material", "uncertain_or_low_confidence", True
    if metallicity["label"] == "mixed":
        reason = "mixed_metallicity"
    elif metallicity["label"] == "metallic" and metallicity["confidence"] == "high":
        word, reason, fallback = "metallic", "high_confidence_metallic", False
    elif roughness["confidence"] in ("high", "medium"):
        if roughness["label"] == "low":
            word, reason, fallback = "glossy", "low_roughness", False
        elif roughness["label"] == "high":
            word, reason, fallback = "matte", "high_roughness", False
        elif roughness["label"] == "medium":
            reason = "medium_roughness_has_no_dedicated_initializer"
    return {"policy_version": POLICY_VERSION, "initializer": word,
            "selection_reason": reason, "fallback": fallback}
