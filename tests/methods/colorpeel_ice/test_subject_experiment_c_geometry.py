"""Verify exact mask projection and material-independent manual geometry."""

import copy
import json

import numpy as np
import pytest

from experiments.subject_albedo_pbr_c.geometry import DEFAULT_GEOMETRY, boundary_uv, build_mesh, frame_padding, surface_depth


def test_each_foreground_pixel_has_exactly_one_unique_positive_facing_quad():
    mask = np.zeros((512, 512), dtype=bool)
    mask[430:440, 205:218] = True
    mask[433:437, 209:212] = False
    mask[199, 111] = True
    mesh = build_mesh(mask, DEFAULT_GEOMETRY)
    vertices, faces = np.asarray(mesh["vertices"]), np.asarray(mesh["faces"])
    pixels = np.column_stack(((vertices[:, 0] + 0.5) * 512, (0.5 - vertices[:, 1]) * 512))
    reconstructed = np.zeros_like(mask, dtype=np.int64)
    for face in faces:
        quad = pixels[face]
        x, y = np.min(quad, axis=0).astype(int)
        assert np.array_equal(quad, [[x, y], [x, y + 1], [x + 1, y + 1], [x + 1, y]])
        reconstructed[y, x] += 1
    assert np.array_equal(reconstructed, mask.astype(int))
    assert len({tuple(face) for face in faces}) == int(mask.sum())
    assert len({tuple(vertex[:2]) for vertex in vertices}) == len(vertices)
    normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                       vertices[faces[:, 2]] - vertices[faces[:, 0]])
    assert np.isfinite(normals).all() and np.all(normals[:, 2] > 0)
    assert mesh["face_count"] == mesh["foreground_pixel_count"] == int(mask.sum())


def test_uv_and_orthographic_projection_are_exact_at_image_edges():
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[0, 0] = mask[-1, -1] = 255
    mesh = build_mesh(mask, DEFAULT_GEOMETRY)
    xyz, uv = np.asarray(mesh["vertices"]), np.asarray(mesh["uv"])
    assert np.array_equal(uv[:, 0], xyz[:, 0] + 0.5)
    assert np.array_equal(uv[:, 1], xyz[:, 1] + 0.5)
    assert uv.min() == 0 and uv.max() == 1
    assert mesh["depth_statistics"]["minimum"] >= 0


def test_depth_has_flat_backplate_curved_body_and_finite_continuous_normals():
    config = DEFAULT_GEOMETRY
    near, far = np.asarray(config["axis_near_px"]), np.asarray(config["axis_far_px"])
    u = (far - near) / np.linalg.norm(far - near)
    perpendicular = np.array([-u[1], u[0]])
    center = near + u * 90
    q = np.linspace(-80, 80, 3201)
    xy = center + q[:, None] * perpendicular
    z = surface_depth(xy[:, 0], xy[:, 1], config)
    assert np.isfinite(z).all()
    assert np.allclose(z[np.abs(q) >= config["radius_px"]], config["backplate_depth"])
    assert z[np.argmin(abs(q))] > config["backplate_depth"] + 0.2
    assert np.all((z >= 0) & (z <= 0.3))
    dz = np.gradient(z, q / 512)
    normals = np.column_stack((-dz, np.zeros_like(dz), np.ones_like(dz)))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    assert np.isfinite(normals).all()
    maximum_jump = np.max(np.linalg.norm(np.diff(normals, axis=0), axis=1))
    assert maximum_jump < 0.15
    finer_q = np.linspace(-80, 80, 6401)
    finer_xy = center + finer_q[:, None] * perpendicular
    finer_z = surface_depth(finer_xy[:, 0], finer_xy[:, 1], config)
    finer_dz = np.gradient(finer_z, finer_q / 512)
    finer_normals = np.column_stack((-finer_dz, np.zeros_like(finer_dz), np.ones_like(finer_dz)))
    finer_normals /= np.linalg.norm(finer_normals, axis=1, keepdims=True)
    assert np.max(np.linalg.norm(np.diff(finer_normals, axis=0), axis=1)) < 0.6 * maximum_jump


def test_cap_to_body_transition_is_continuous_and_depth_is_config_driven():
    config = copy.deepcopy(DEFAULT_GEOMETRY)
    near, far = np.asarray(config["axis_near_px"]), np.asarray(config["axis_far_px"])
    u = (far - near) / np.linalg.norm(far - near)
    perpendicular = np.array([-u[1], u[0]])
    for q in (-45, -25, 0, 25, 45):
        arc = np.sqrt(1.0 - (q / config["radius_px"]) ** 2)
        boundary = config["cap_projection_radius_px"] * arc + config["cap_boundary_offset_px"]
        t = np.linspace(boundary - 4, boundary + 4, 1601)
        xy = near + t[:, None] * u + q * perpendicular
        depth = surface_depth(xy[:, 0], xy[:, 1], config)
        assert np.max(np.abs(np.diff(depth))) < 0.0001
        expected = config["backplate_depth"] + config["axis_depth_tilt"] * (
            1 - boundary / np.linalg.norm(far - near)) + config["body_height"] * arc
        assert depth[len(depth) // 2] == pytest.approx(expected, abs=1e-12)
        derivative = np.gradient(depth, t)
        assert abs(derivative[800] - derivative[799]) < 2e-5
    config["body_height"] = 0.18
    assert surface_depth(near[0] + u[0] * 90, near[1] + u[1] * 90, config) != surface_depth(
        near[0] + u[0] * 90, near[1] + u[1] * 90, DEFAULT_GEOMETRY)


def test_elliptical_join_is_closer_to_the_observed_cap_edge_than_old_straight_join():
    config = DEFAULT_GEOMETRY
    near, far = np.asarray(config["axis_near_px"]), np.asarray(config["axis_far_px"])
    u = (far - near) / np.linalg.norm(far - near)
    perpendicular = np.array([-u[1], u[0]])
    q = np.linspace(-config["radius_px"], config["radius_px"], 10001)
    t = config["cap_projection_radius_px"] * np.sqrt(1 - (q / config["radius_px"]) ** 2)
    t += config["cap_boundary_offset_px"]
    curve = near + t[:, None] * u + q[:, None] * perpendicular
    landmarks = np.array([[191, 441], [221, 451], [247, 478], [260, 511]])
    fitted_distances = np.array([np.linalg.norm(curve - point, axis=1).min() for point in landmarks])
    # The old 0..25px transition's center was a straight t=12.5 join.
    old_distances = np.abs((landmarks - near) @ u - 12.5)
    assert np.all(fitted_distances < old_distances)
    assert fitted_distances.max() < 12 and fitted_distances.mean() < 5


def test_negative_depth_origin_does_not_change_projection_faces_or_uv():
    mask = np.zeros((512, 512), dtype=bool)
    mask[477:507, 180:225] = True
    mesh = build_mesh(mask, DEFAULT_GEOMETRY)
    moved = build_mesh(mask, {**DEFAULT_GEOMETRY, "backplate_depth": -1.0})
    assert mesh["depth_statistics"]["minimum"] < 0
    assert moved["depth_statistics"]["minimum"] < -1
    assert mesh["faces"] == moved["faces"] and mesh["uv"] == moved["uv"]
    assert np.array_equal(np.asarray(mesh["vertices"])[:, :2], np.asarray(moved["vertices"])[:, :2])
    assert np.allclose(np.asarray(moved["vertices"])[:, 2] - np.asarray(mesh["vertices"])[:, 2], -1.02)


def test_json_serializable_and_identical_across_all_material_conditions():
    assert json.loads(json.dumps(DEFAULT_GEOMETRY)) == DEFAULT_GEOMETRY
    mask = np.zeros((512, 512), dtype=bool)
    mask[410:420, 240:250] = True
    first = build_mesh(mask, DEFAULT_GEOMETRY)
    second = build_mesh(mask, json.loads(json.dumps(DEFAULT_GEOMETRY)))
    assert first == second
    assert json.loads(json.dumps(first)) == first
    assert "not ground-truth" in first["fit_description"]
    for name in ("albedo", "metallic", "roughness"):
        with pytest.raises(ValueError, match="only"):
            build_mesh(mask, {**DEFAULT_GEOMETRY, name: 0.5})


@pytest.mark.parametrize("mask", [np.zeros((512, 512)), np.ones((10, 10)),
                                  np.full((512, 512), 0.5), np.full((512, 512), np.nan)])
def test_invalid_masks_are_rejected(mask):
    with pytest.raises(ValueError, match="binary"):
        build_mesh(mask, DEFAULT_GEOMETRY)


@pytest.mark.parametrize("change", [{"radius_px": 0}, {"cap_blend_px": 0},
                                   {"cap_projection_radius_px": 0},
                                   {"body_height": float("nan")}, {"image_size": 256},
                                   {"axis_far_px": [207, 500]}, {"top_transition_px": 100}])
def test_invalid_geometry_is_rejected(change):
    with pytest.raises(ValueError):
        surface_depth(200, 440, {**DEFAULT_GEOMETRY, **change})


@pytest.fixture
def repaired_uv_inputs():
    base = np.zeros((512, 512), dtype=np.uint8)
    base[210:213, 220:223] = 255
    repaired = base.copy()
    repaired[210:214, 220:224] = 255
    y, x = np.indices(base.shape)
    nearest = np.stack((np.clip(x, 220, 222), np.clip(y, 210, 212)), axis=2)
    return build_mesh(repaired, DEFAULT_GEOMETRY), base, nearest


def test_boundary_uv_preserves_every_original_face_exactly(repaired_uv_inputs):
    mesh, base, nearest = repaired_uv_inputs
    old_uv = np.asarray(mesh["uv"])[np.asarray(mesh["faces"])]
    result, stats = boundary_uv(mesh, base, nearest)
    vertices = np.asarray(mesh["vertices"])
    centers = vertices[np.asarray(mesh["faces"])].mean(axis=1)
    pixels = np.column_stack(((centers[:, 0] + 0.5) * 512, (0.5 - centers[:, 1]) * 512)).astype(int)
    original_faces = base[pixels[:, 1], pixels[:, 0]] > 0
    assert np.array_equal(result[original_faces], old_uv[original_faces])
    assert stats["original_face_count"] == 9 and stats["render_face_count"] == 16
    assert stats["added_face_count"] == 7 and stats["added_face_ratio"] == 7 / 16
    assert stats["max_pixel_extension"] == pytest.approx(np.sqrt(2))


def test_added_faces_sample_only_original_foreground_pixel_centers(repaired_uv_inputs):
    mesh, base, nearest = repaired_uv_inputs
    old_uv = np.asarray(mesh["uv"])[np.asarray(mesh["faces"])]
    result, _ = boundary_uv(mesh, base, nearest)
    changed = np.any(result != old_uv, axis=(1, 2))
    assert changed.sum() == 7
    assert np.all(result[changed] == result[changed, :1, :])
    xy = np.stack((result[changed, 0, 0] * 512 - 0.5,
                   (1 - result[changed, 0, 1]) * 512 - 0.5), axis=1)
    assert np.array_equal(xy, np.rint(xy))
    xy = xy.astype(int)
    assert np.all(base[xy[:, 1], xy[:, 0]] == 255)


def test_boundary_uv_is_material_independent_and_does_not_mutate_inputs(repaired_uv_inputs):
    import inspect

    mesh, base, nearest = repaired_uv_inputs
    before = json.dumps(mesh)
    original_mask, original_nearest = base.copy(), nearest.copy()
    first, stats = boundary_uv(mesh, base, nearest)
    second, second_stats = boundary_uv(mesh, base, nearest)
    assert list(inspect.signature(boundary_uv).parameters) == ["mesh", "base_mask", "nearest_xy"]
    assert np.array_equal(first, second) and stats == second_stats
    assert json.dumps(mesh) == before
    assert np.array_equal(base, original_mask) and np.array_equal(nearest, original_nearest)
    assert "not recovered Albedo" in stats["description"]
    assert json.loads(json.dumps(stats)) == stats


@pytest.mark.parametrize("kind", ["outside_foreground", "out_of_bounds", "noninteger", "wrong_shape"])
def test_boundary_uv_rejects_invalid_nearest(repaired_uv_inputs, kind):
    mesh, base, nearest = repaired_uv_inputs
    if kind == "outside_foreground":
        nearest[0, 0] = [0, 0]
    elif kind == "out_of_bounds":
        nearest[0, 0] = [512, 0]
    elif kind == "noninteger":
        nearest = nearest.astype(float)
    else:
        nearest = nearest[:, :, 0]
    with pytest.raises(ValueError, match="nearest"):
        boundary_uv(mesh, base, nearest)


def test_boundary_uv_rejects_empty_base_or_missing_original_coverage(repaired_uv_inputs):
    mesh, base, nearest = repaired_uv_inputs
    with pytest.raises(ValueError, match="nonempty"):
        boundary_uv(mesh, np.zeros_like(base), nearest)
    incomplete = copy.deepcopy(mesh)
    incomplete["faces"].pop(0)
    with pytest.raises(ValueError, match="cover"):
        boundary_uv(incomplete, base, nearest)


@pytest.fixture
def bottom_padding_inputs():
    mask = np.zeros((512, 512), dtype=bool)
    mask[509:512, 200:204] = True
    mesh = build_mesh(mask, DEFAULT_GEOMETRY)
    face_uv = np.asarray(mesh["uv"])[np.asarray(mesh["faces"])].copy()
    face_uv[-4:] = np.array([0.4, 0.01])
    return mesh, face_uv, mask


def test_frame_padding_preserves_visible_geometry_and_authoritative_uv_prefix(bottom_padding_inputs):
    mesh, uv, mask = bottom_padding_inputs
    before = json.dumps(mesh)
    old_uv = uv.copy()
    padded, padded_uv, stats = frame_padding(mesh, uv, mask, DEFAULT_GEOMETRY)
    assert padded["vertices"][:len(mesh["vertices"])] == mesh["vertices"]
    assert padded["faces"][:len(mesh["faces"])] == mesh["faces"]
    assert padded["uv"][:len(mesh["uv"])] == mesh["uv"]
    assert np.array_equal(padded_uv[:len(uv)], uv)
    assert json.dumps(mesh) == before and np.array_equal(old_uv, uv)
    assert stats["padded_faces"] == 4 and stats["outside_strip_pixels"] == 8
    assert stats["added_vertices"] == 5
    assert padded["vertex_count"] == mesh["vertex_count"] + 5
    assert padded["face_count"] == mesh["face_count"] + 4


def test_padded_faces_are_outside_frame_shared_and_positive_facing(bottom_padding_inputs):
    mesh, uv, mask = bottom_padding_inputs
    padded, padded_uv, _ = frame_padding(mesh, uv, mask, DEFAULT_GEOMETRY)
    vertices = np.asarray(padded["vertices"])
    added = np.asarray(padded["faces"])[len(mesh["faces"]):]
    coordinates = vertices[added]
    pixel_y = (0.5 - coordinates[:, :, 1]) * 512
    assert np.all(pixel_y >= 512) and set(pixel_y.ravel()) == {512, 514}
    assert np.array_equal(added[:-1, 2], added[1:, 1])
    assert np.array_equal(added[:-1, 3], added[1:, 0])
    normals = np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0])
    assert np.all(normals[:, 2] > 0)
    assert np.all(padded_uv[len(uv):] == np.array([0.4, 0.01]))
    assert np.all(np.asarray(padded["uv"])[len(mesh["uv"]):, 1] < 0)


def test_frame_padding_is_deterministic_and_has_no_material_inputs(bottom_padding_inputs):
    import hashlib
    import inspect

    mesh, uv, mask = bottom_padding_inputs
    first, first_uv, stats = frame_padding(mesh, uv, mask, DEFAULT_GEOMETRY)
    second, second_uv, other_stats = frame_padding(mesh, uv, mask, DEFAULT_GEOMETRY)
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    assert digest(first) == digest(second) and first_uv.tobytes() == second_uv.tobytes()
    assert stats == other_stats
    assert list(inspect.signature(frame_padding).parameters) == ["mesh", "face_uv", "render_mask", "config", "pixels"]


def test_frame_padding_rejects_other_frame_edges(bottom_padding_inputs):
    _, _, mask = bottom_padding_inputs
    mask = mask.copy()
    mask[0, 200] = True
    mesh = build_mesh(mask, DEFAULT_GEOMETRY)
    uv = np.asarray(mesh["uv"])[np.asarray(mesh["faces"])]
    with pytest.raises(ValueError, match="bottom-frame contact only"):
        frame_padding(mesh, uv, mask, DEFAULT_GEOMETRY)


def test_frame_padding_with_no_bottom_contact_does_not_add_geometry():
    mask = np.zeros((512, 512), dtype=bool)
    mask[400:402, 200:203] = True
    mesh = build_mesh(mask, DEFAULT_GEOMETRY)
    uv = np.asarray(mesh["uv"])[np.asarray(mesh["faces"])]
    padded, padded_uv, stats = frame_padding(mesh, uv, mask, DEFAULT_GEOMETRY)
    assert padded["vertices"] == mesh["vertices"] and padded["faces"] == mesh["faces"]
    assert np.array_equal(padded_uv, uv)
    assert stats["padded_faces"] == stats["outside_strip_pixels"] == 0
