"""Render a continuous gray studio without changing the frozen foreground recipe."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.methods.colorpeel_ice import render_d1_color_calibration_preflight as shared
from scripts.methods.colorpeel_ice import render_d1_color_calibration_reduced_direct as direct
from scripts.methods.colorpeel_ice import run_d1_gamut_aware_lch_injection as original

SOURCE = Path('/home/r12user5/Documents/Jiawei/colorpeel-runs/natural_image_subject_color_pilot/gamut_aware_lch_injection_f1af34e_20260912_0080')
CONTRACT_SHA = 'cc703f95dbcd9d10345283fb16f21903132129b757dd6bc0dd079d1240649cc8'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


def rebuild_ground(scene):
    """12-unit floor, tangent 8-unit cove, radius-20 wall up to z=32."""
    bpy = shared.bpy
    old = bpy.data.objects['Ground']
    evidence = {'location': list(old.location), 'matrix_world': [list(r) for r in old.matrix_world],
                'bounds_local': [list(v) for v in old.bound_box], 'vertices': len(old.data.vertices)}
    material = old.data.materials[0]
    bpy.data.objects.remove(old, do_unlink=True)
    rings = [(12., 0.)]
    rings += [(12 + 8 * math.sin(t), 8 * (1 - math.cos(t)))
              for t in np.linspace(0, math.pi / 2, 65)[1:]]
    rings.append((20., 32.))
    segments = 256
    vertices = [(r * math.cos(2 * math.pi * i / segments),
                 r * math.sin(2 * math.pi * i / segments), z)
                for r, z in rings for i in range(segments)]
    # Upward floor normal and inward wall normals; tangent joins, no horizon edge.
    faces = [tuple(range(segments))]
    for j in range(len(rings) - 1):
        for i in range(segments):
            k = (i + 1) % segments
            faces.append((j * segments + i, (j + 1) * segments + i,
                          (j + 1) * segments + k, j * segments + k))
    mesh = bpy.data.meshes.new('ContinuousGrayStudio')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    ground = bpy.data.objects.new('Ground', mesh)
    scene.collection.objects.link(ground)
    ground.data.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    bpy.context.view_layer.update()
    return {'old_geometry': evidence, 'floor_radius': 12, 'cove_radius': 8,
            'wall_radius': 20, 'wall_height': 32, 'radial_segments': segments,
            'cove_segments': 64, 'background_pixel_replacement': False}


def coverage(scene):
    """Check all 512x512 camera rays against the background alone."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    ground = shared.bpy.data.objects['Ground']
    tree = BVHTree.FromObject(ground, shared.bpy.context.evaluated_depsgraph_get())
    camera = scene.camera
    frame = camera.data.view_frame(scene=scene)
    xs, ys = [v.x for v in frame], [v.y for v in frame]
    rotation = camera.matrix_world.to_3x3()
    missed = 0
    distances = []
    # Include frame boundary as well as every output pixel center.
    points = [(v.x, v.y) for v in frame]
    points.extend((min(xs) + (x + .5) / 512 * (max(xs) - min(xs)),
                   min(ys) + (y + .5) / 512 * (max(ys) - min(ys)))
                  for y in range(512) for x in range(512))
    for x, y in points:
        direction = (rotation @ Vector((x, y, frame[0].z))).normalized()
        hit = tree.ray_cast(camera.location, direction, camera.data.clip_end)
        if hit[0] is None:
            missed += 1
        else:
            distances.append(hit[3])
    if missed:
        raise ValueError(f'{missed} camera rays expose World instead of studio')
    return {'rays': len(points), 'missed_background': missed,
            'distance_min_max': [min(distances), max(distances)],
            'clip_start_end': [camera.data.clip_start, camera.data.clip_end]}


def run(output, asset_root, shapes):
    bpy = shared.bpy
    output = output.resolve()
    if output.exists() or output.is_relative_to(ROOT) or output.is_relative_to(SOURCE):
        raise ValueError('require fresh external output')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('checkout must be clean')
    contract_path = SOURCE / original.CONTRACT_NAME
    if shared.file_sha256(contract_path) != CONTRACT_SHA:
        raise ValueError('historical asset contract changed')
    contract = json.loads(contract_path.read_text())
    manifest_path = SOURCE / original.NEUTRAL_MANIFEST_NAME
    manifest_sha = '04d843102f0b5a8ade7c2f74f8af32a6dc406448d49522a0b12a23454a06fa4b'
    if shared.file_sha256(manifest_path) != manifest_sha:
        raise ValueError('historical render manifest changed')
    records = {r['request_id']: r for r in json.loads(manifest_path.read_text())['records']}
    paths = original._runtime_assets(asset_root, contract)
    locks = {str(contract_path): CONTRACT_SHA, str(manifest_path): manifest_sha}
    locks.update({str(paths[k]): v['sha256'] for k, v in contract['assets'].items()})
    output.mkdir(parents=True)
    write(output / 'protocol.json', {'git': shared._git_commit(), 'shapes': shapes,
          'views': [0, 8, 16], 'samples': 512, 'render_seed': 42,
          'training_authorized': False, 'source': str(SOURCE),
          'change': 'replace Ground geometry only; no background pixel editing'})
    for request in original.injection.neutral_requests():
        if request['shape'] not in shapes:
            continue
        case = output / request['request_id']
        case.mkdir()
        source = SOURCE / 'neutral' / 'renders' / request['request_id']
        for name in ('metadata.json', 'image.png', 'mask.png'):
            locks[str(source / name)] = shared.file_sha256(source / name)
            if locks[str(source / name)] != records[request['request_id']][name.split('.')[0] + '_sha256']:
                raise ValueError('frozen source hash mismatch')
        old = json.loads((source / 'metadata.json').read_text())
        for kind in ('image', 'mask'):
            if locks[str(source / (kind + '.png'))] != old[kind + '_sha256']:
                raise ValueError('historical render hash mismatch')
        bpy.ops.wm.open_mainfile(filepath=str(paths['base_scene']))
        scene = bpy.context.scene
        devices = shared._configure_cycles(scene, request['render_seed'])
        cm = shared.configure_color_management(scene)
        for obj in list(scene.objects):
            if obj.type == 'MESH' and obj.name != 'Ground':
                bpy.data.objects.remove(obj, do_unlink=True)
        world, ground = shared._configure_background(scene)
        geometry = rebuild_ground(scene)
        asset_name = original.SHAPE_DETAILS[request['shape']][0]
        obj = original._append_shape(request['shape'], paths[asset_name])
        linear = original.injection.lab_to_linear_rgb(np.array([original.injection.NEUTRAL_LAB]))[0].tolist()
        material = direct._configure_rubber(obj, {'material': 'Rubber', 'linear_rgb': linear, 'socket_rgba': linear + [1.]}, paths['material_asset'])
        camera, lights, base = direct._camera_and_lights(obj, request['view_index'])
        object_state = {'shape': request['shape'], 'location': list(obj.location),
                        'scale': list(obj.scale), 'rotation_euler': list(obj.rotation_euler)}
        retained = {'camera': camera, 'lights': lights, 'base_scene_state': base,
                    'object': object_state, 'material': material, 'world': world,
                    'ground': ground, 'color_management': cm, 'render_seed': scene.cycles.seed}
        for key, value in retained.items():
            if value != old[key]:
                raise ValueError(f'frozen state differs: {key}')
        covered = coverage(scene)
        shared._configure_mask_output(case, obj)
        scene.render.filepath = str(case / 'image.png')
        bpy.ops.wm.save_as_mainfile(filepath=str(case / 'scene.blend'))
        bpy.ops.render.render(write_still=True)
        shared._finalize_mask(case)
        write(case / 'metadata.json', {**retained, 'request': request, 'geometry': geometry,
              'coverage': covered, 'cuda_devices': devices, 'blender_version': list(bpy.app.version),
              'source_metadata': str(source / 'metadata.json'), 'frozen_state_equal': True,
              'world_strength': float(next(n for n in scene.world.node_tree.nodes if n.type == 'BACKGROUND').inputs['Strength'].default_value)})
    for path, digest in locks.items():
        if shared.file_sha256(Path(path)) != digest:
            raise ValueError(f'input changed during rendering: {path}')
    write(output / 'source_locks.json', locks)
    write(output / 'files.json', {str(p.relative_to(output)): shared.file_sha256(p)
                                for p in output.rglob('*') if p.is_file()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--shapes', nargs='+', choices=['sphere', 'cube', 'cylinder'], required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    run(args.output, args.asset_root, args.shapes)
