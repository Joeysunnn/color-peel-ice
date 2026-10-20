"""Manually fitted, photo-aligned mailbox heightfield; not ground-truth geometry."""

import copy

import numpy as np


DEFAULT_GEOMETRY = {
    "image_size": 512,
    "axis_near_px": [207.0, 500.0],
    "axis_far_px": [397.0, 439.0],
    "radius_px": 64.0,
    "backplate_depth": 0.02,
    "body_height": 0.22,
    "axis_depth_tilt": 0.035,
    "top_transition_px": 8.0,
    "cap_projection_radius_px": 45.0,
    "cap_boundary_offset_px": -5.0,
    "cap_blend_px": 3.0,
}


def _settings(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULT_GEOMETRY):
        raise ValueError("geometry accepts only the declared manual-fit parameters")
    settings = copy.deepcopy({**DEFAULT_GEOMETRY, **config})
    if type(settings["image_size"]) is not int or settings["image_size"] != 512:
        raise ValueError("manual mailbox fit requires the original 512px projection")
    near, far = (np.asarray(settings[name], dtype=np.float64)
                 for name in ("axis_near_px", "axis_far_px"))
    if (near.shape != (2,) or far.shape != (2,) or not np.isfinite([near, far]).all()
            or np.linalg.norm(far - near) <= 0):
        raise ValueError("mailbox axis requires two distinct finite pixel coordinates")
    numeric = set(DEFAULT_GEOMETRY) - {"image_size", "axis_near_px", "axis_far_px"}
    if any(isinstance(settings[name], bool) or not np.isscalar(settings[name])
           or not np.isfinite(settings[name]) for name in numeric):
        raise ValueError("manual-fit parameters must be finite numbers")
    if (settings["radius_px"] <= 0 or settings["cap_projection_radius_px"] <= 0
            or settings["cap_blend_px"] <= 0
            or not 0 < settings["top_transition_px"] <= settings["radius_px"]
            or settings["body_height"] < 0):
        raise ValueError("manual-fit radii, transitions and depth amplitudes are invalid")
    return settings


def _smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def surface_depth(x, y, config):
    """Return scene Z from pixel corners, using no image or material values."""
    settings = _settings(config)
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("pixel coordinates must be finite")
    near = np.asarray(settings["axis_near_px"], dtype=np.float64)
    direction = np.asarray(settings["axis_far_px"], dtype=np.float64) - near
    length = np.linalg.norm(direction)
    u = direction / length
    t = (x - near[0]) * u[0] + (y - near[1]) * u[1]
    q = -(x - near[0]) * u[1] + (y - near[1]) * u[0]
    radius = settings["radius_px"]
    relative_q = np.clip(q / radius, -1.0, 1.0)
    arc = np.sqrt(np.maximum(0.0, 1.0 - relative_q * relative_q))
    # Smooth the arc into the backplate at the silhouette instead of retaining
    # an infinite derivative at |q|=radius. The original mask defines visibility.
    edge = _smoothstep((radius - np.abs(q)) / settings["top_transition_px"])
    tilt = settings["axis_depth_tilt"] * (1.0 - np.clip(t / length, 0.0, 1.0))
    cylinder = tilt + settings["body_height"] * arc
    # The visible cap edge is fitted as an ellipse in image coordinates.
    # Its plane meets the cylindrical surface at that ellipse, instead of a
    # straight constant-t join. This is an explicit manual single-view fit.
    boundary = settings["cap_projection_radius_px"] * arc + settings["cap_boundary_offset_px"]
    cap = tilt + settings["body_height"] * (
        t - settings["cap_boundary_offset_px"]) / settings["cap_projection_radius_px"]
    # cap_blend_px is the half-width of the smooth join around the fitted edge.
    cylinder_weight = _smoothstep((t - boundary + settings["cap_blend_px"]) /
                                   (2.0 * settings["cap_blend_px"]))
    return settings["backplate_depth"] + edge * (
        cylinder_weight * cylinder + (1.0 - cylinder_weight) * cap)


def build_mesh(mask, config):
    """Make one +Z-facing quad per foreground pixel with exact photo UVs."""
    settings = _settings(config)
    mask = np.asarray(mask)
    if (mask.shape != (512, 512) or not np.isin(mask, [0, 1, 255]).all()
            or not np.any(mask)):
        raise ValueError("mesh requires a nonempty binary 512px object mask")
    foreground = mask.astype(bool)
    corners = np.zeros((513, 513), dtype=bool)
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        corners[dy:dy + 512, dx:dx + 512] |= foreground
    ys, xs = np.nonzero(corners)
    ids = np.full(corners.shape, -1, dtype=np.int64)
    ids[ys, xs] = np.arange(len(xs))
    z = surface_depth(xs, ys, settings)
    vertices = np.column_stack((xs / 512.0 - 0.5, 0.5 - ys / 512.0, z))
    uv = np.column_stack((xs / 512.0, 1.0 - ys / 512.0))
    fy, fx = np.nonzero(foreground)
    faces = np.column_stack((ids[fy, fx], ids[fy + 1, fx],
                             ids[fy + 1, fx + 1], ids[fy, fx + 1]))
    return {
        "vertices": vertices.tolist(), "faces": faces.tolist(), "uv": uv.tolist(),
        "foreground_pixel_count": int(foreground.sum()),
        "vertex_count": len(vertices), "face_count": len(faces),
        "depth_statistics": {"minimum": float(z.min()), "maximum": float(z.max()),
                             "mean": float(z.mean()), "std": float(z.std())},
        "geometry_config": settings,
        "coordinate_system": "pixel (x,y) -> (x/512-0.5, 0.5-y/512, manual depth); camera (0,0,4), ortho scale 1",
        "fit_description": "manual single-view mailbox cylinder/elliptical-cap/backplate heightfield; not ground-truth or recovered 3D; no hidden surfaces",
    }
