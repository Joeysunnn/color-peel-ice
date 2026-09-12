from __future__ import annotations

import numpy as np

from src.methods.colorpeel_ice import natural_subject_recolor_border_repair as repair


def test_bottom_border_repair_adds_only_the_approved_missing_pixels():
    raw = np.zeros((512, 512), dtype=np.uint8)
    raw[504:508, 180:459] = 255
    raw[508, 250:459] = 255
    raw[509, 389:459] = 255
    spec = {
        "rectangle_xyxy_inclusive": [180, 508, 458, 511],
        "expected_added_pixel_count": 837,
        "pre_repair_mask_runs": {
            "504": [[180, 458]], "505": [[180, 458]], "506": [[180, 458]], "507": [[180, 458]],
            "508": [[250, 458]], "509": [[389, 458]], "510": [], "511": [],
        },
    }
    repaired, added = repair.repair_bottom_border(raw, spec)
    assert int(added.sum()) == 837
    assert np.all(repaired[508:512, 180:459])
    assert np.all(repaired[raw == 255])


def test_repaired_alpha_is_zero_outside_and_opaque_at_the_canvas_bottom_interior():
    raw = np.zeros((512, 512), dtype=np.uint8)
    raw[500:512, 180:459] = 255
    alpha, metadata = repair.derive_repaired_alpha(raw == 255)
    assert metadata["w_out_px"] == 0
    assert np.all(alpha[raw == 0] == 0)
    assert alpha[511, 300] == 65535
