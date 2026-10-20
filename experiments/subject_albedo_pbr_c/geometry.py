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


def boundary_uv(mesh, base_mask, nearest_xy):
    """Extend UVs only on repaired-mask pixels, without editing any Albedo."""
    base = np.asarray(base_mask)
    nearest = np.asarray(nearest_xy)
    if base.shape != (512, 512) or not np.isin(base, [0, 1, 255]).all() or not np.any(base):
        raise ValueError("UV continuation requires a nonempty binary original mask")
    base = base.astype(bool)
    if (nearest.shape != (512, 512, 2) or not np.issubdtype(nearest.dtype, np.integer)
            or np.any(nearest < 0) or np.any(nearest >= 512)
            or not np.all(base[nearest[:, :, 1], nearest[:, :, 0]])):
        raise ValueError("nearest XY must contain integer indices into original-mask foreground")
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    faces = np.asarray(mesh["faces"])
    uv = np.asarray(mesh["uv"], dtype=np.float64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 4 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer) or np.any(faces < 0)
            or np.any(faces >= len(vertices)) or uv.shape != (len(vertices), 2)
            or not np.isfinite(uv).all()):
        raise ValueError("UV continuation requires the valid pixel-quad mesh")
    projected = np.stack(((vertices[:, 0] + 0.5) * 512,
                           (0.5 - vertices[:, 1]) * 512), axis=1)[faces]
    pixels = np.rint(projected.min(axis=1)).astype(np.int64)
    expected = pixels[:, None, :] + np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    if (not np.allclose(projected, expected, atol=1e-7, rtol=0)
            or np.any(pixels < 0) or np.any(pixels >= 512)
            or len(np.unique(pixels, axis=0)) != len(pixels)):
        raise ValueError("render faces must be unique exact image-pixel quads")
    covered = np.zeros_like(base)
    covered[pixels[:, 1], pixels[:, 0]] = True
    if not np.all(covered[base]):
        raise ValueError("render faces must cover every original-mask pixel")
    added = ~base[pixels[:, 1], pixels[:, 0]]
    face_uv = uv[faces].copy()
    origins = pixels[added]
    sources = nearest[origins[:, 1], origins[:, 0]]
    centers = np.column_stack(((sources[:, 0] + 0.5) / 512,
                                1 - (sources[:, 1] + 0.5) / 512))
    face_uv[added] = centers[:, None, :]
    distances = np.linalg.norm(sources - origins, axis=1)
    return face_uv, {
        "original_face_count": int(base.sum()), "render_face_count": len(faces),
        "added_face_count": int(added.sum()), "added_face_ratio": float(added.mean()),
        "max_pixel_extension": float(distances.max()) if len(distances) else 0.0,
        "description": "Only repaired-mask faces use fixed nearest original-mask pixel-center UVs; boundary continuation, not recovered Albedo; original Albedo files and original-face UVs are unchanged.",
    }


def frame_padding(mesh, face_uv, render_mask, config, pixels=2):
    """Continue bottom-cut geometry outside the frame, keeping visible data exact."""
    settings = _settings(config)
    mask = np.asarray(render_mask)
    if (type(pixels) is not int or pixels <= 0 or mask.shape != (512, 512)
            or not np.isin(mask, [0, 1, 255]).all() or not np.any(mask)):
        raise ValueError("bottom-frame padding requires a binary mask and positive integer pixels")
    mask = mask.astype(bool)
    if mask[0].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError("this manual fit supports bottom-frame contact only")
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    faces = np.asarray(mesh["faces"])
    uv = np.asarray(mesh["uv"], dtype=np.float64)
    face_uv = np.asarray(face_uv, dtype=np.float64)
    if (mesh.get("geometry_config") != settings or vertices.ndim != 2 or vertices.shape[1] != 3
            or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 4
            or not np.issubdtype(faces.dtype, np.integer) or np.any(faces < 0)
            or np.any(faces >= len(vertices)) or uv.shape != (len(vertices), 2)
            or face_uv.shape != (len(faces), 4, 2) or not np.isfinite(uv).all()
            or not np.isfinite(face_uv).all()):
        raise ValueError("frame padding requires the matching fixed mesh and per-face UVs")
    projected = np.stack(((vertices[:, 0] + 0.5) * 512,
                           (0.5 - vertices[:, 1]) * 512), axis=1)[faces]
    face_pixels = np.rint(projected.min(axis=1)).astype(np.int64)
    expected = face_pixels[:, None, :] + np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    if (not np.allclose(projected, expected, atol=1e-7, rtol=0)
            or np.any(face_pixels < 0) or np.any(face_pixels >= 512)
            or len(np.unique(face_pixels, axis=0)) != len(face_pixels)):
        raise ValueError("frame padding requires unique visible pixel quads")
    covered = np.zeros_like(mask)
    covered[face_pixels[:, 1], face_pixels[:, 0]] = True
    if not np.array_equal(covered, mask):
        raise ValueError("visible mesh coverage differs from the render mask")
    bottom_faces = {int(x): number for number, (x, y) in enumerate(face_pixels) if y == 511}
    added_vertices, added_uv, added_faces, added_face_uv = [], [], [], []
    outside_ids = {}
    for x in np.flatnonzero(mask[-1]):
        for corner_x in (int(x), int(x) + 1):
            if corner_x not in outside_ids:
                outside_ids[corner_x] = len(vertices) + len(added_vertices)
                outside_y = 512 + pixels
                z = float(surface_depth(corner_x, outside_y, settings))
                added_vertices.append([corner_x / 512 - 0.5, 0.5 - outside_y / 512, z])
                added_uv.append([corner_x / 512, 1 - outside_y / 512])
        number = bottom_faces[int(x)]
        added_faces.append([int(faces[number, 1]), outside_ids[int(x)],
                            outside_ids[int(x) + 1], int(faces[number, 2])])
        added_face_uv.append(np.broadcast_to(face_uv[number].mean(axis=0), (4, 2)).tolist())
    result = copy.deepcopy(mesh)
    result["vertices"] = vertices.tolist() + added_vertices
    result["faces"] = faces.tolist() + added_faces
    result["uv"] = uv.tolist() + added_uv
    result["vertex_count"], result["face_count"] = len(result["vertices"]), len(result["faces"])
    z = np.asarray(result["vertices"])[:, 2]
    result["depth_statistics"] = {"minimum": float(z.min()), "maximum": float(z.max()),
                                  "mean": float(z.mean()), "std": float(z.std())}
    stats = {"edge": "bottom", "padding_pixels": pixels, "padded_faces": len(added_faces),
             "outside_strip_pixels": len(added_faces) * pixels,
             "added_vertices": len(added_vertices), "original_visible_faces": len(faces),
             "description": "Fixed bottom-cut geometry continuation outside the image; visible vertices/faces/UVs are unchanged; added faces use their original bottom-face mean UV; no Albedo or material input is read."}
    result["frame_padding"] = stats
    result["fit_description"] += f"; bottom frame-cut geometry continued {pixels}px outside the image"
    combined_uv = np.concatenate((face_uv, np.asarray(added_face_uv).reshape((-1, 4, 2))), axis=0)
    return result, combined_uv, stats
