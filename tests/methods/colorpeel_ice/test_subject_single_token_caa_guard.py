from pathlib import Path


def test_single_token_caa_has_no_neighbor_fallback_and_requires_zero_weight():
    source = (Path(__file__).parents[3] / "src" / "train" / "train_colorpeel.py").read_text(encoding="utf-8")
    assert "cos_weight must be 0 when fewer than two modifier tokens are learned" in source
    assert "cos = _compute_cosine(attention_maps, indices) if len(indices) >= 2 else torch.zeros((), device=loss.device)" in source
    assert "indices.append(indices[0] + 1)" not in source


def test_subject_only_pilot_disables_adamw_decay_on_masked_base_embeddings():
    config = (
        Path(__file__).parents[3]
        / "experiments"
        / "natural_image_subject_color_pilot"
        / "configs"
        / "d1_subject_recolor_gorilla_short100.yaml"
    ).read_text(encoding="utf-8")
    assert "adam_weight_decay: 0.0" in config
