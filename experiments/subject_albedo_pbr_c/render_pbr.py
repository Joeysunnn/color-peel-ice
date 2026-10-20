"""Blender-only fixed-view mailbox surface rendering with real Cycles/Principled."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def setup(plan, output):
    protocol = plan["protocol"]
    if tuple(protocol["renderer"]["version"]) != bpy.app.version:
        raise ValueError("frozen Blender version differs")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = protocol["renderer"]["samples"]
    scene.cycles.seed = protocol["renderer"]["seed"]
    scene.cycles.use_denoising = False
    scene.cycles.device = "GPU"
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "CUDA"
    preferences.get_devices()
    devices = [device for device in preferences.devices if device.type == "CUDA"]
    if len(devices) != 1:
        raise ValueError("one isolated CUDA device is required")
    for device in preferences.devices:
        device.use = device.type == "CUDA"
    scene.render.resolution_x = scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.exr_codec = "ZIP"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    world = bpy.data.worlds.new("Fixed_assumed_environment")
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs["Color"].default_value = (1, 1, 1, 1)
    world.node_tree.nodes.get("Background").inputs["Strength"].default_value = protocol["world_strength"]
    scene.world = world
    camera_data = bpy.data.cameras.new("Fixed_photo_camera")
    camera = bpy.data.objects.new(camera_data.name, camera_data)
    scene.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = protocol["camera"]["ortho_scale"]
    camera.location = protocol["camera"]["location"]
    camera.rotation_euler = protocol["camera"]["rotation"]
    scene.camera = camera
    for spec in protocol["lights"]:
        light_data = bpy.data.lights.new(spec["name"], "AREA")
        light_data.shape = "RECTANGLE"
        light_data.energy = spec["energy"]
        light_data.size, light_data.size_y = spec["size"], spec["size_y"]
        light_data.color = spec["color"]
        light = bpy.data.objects.new(spec["name"], light_data)
        scene.collection.objects.link(light)
        light.location = spec["location"]
        light.rotation_euler = (Vector(spec["target"]) - light.location).to_track_quat("-Z", "Y").to_euler()
    with np.load(plan["mesh"], allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in ("vertices", "faces", "uv")}
    mesh = bpy.data.meshes.new("Photo_aligned_mailbox_surface")
    mesh.from_pydata(arrays["vertices"].tolist(), [], arrays["faces"].tolist())
    mesh.update()
    uv = mesh.uv_layers.new(name="Fixed_image_projection")
    for polygon in mesh.polygons:
        polygon.use_smooth = True
        for index in polygon.loop_indices:
            uv.data[index].uv = arrays["uv"][mesh.loops[index].vertex_index]
    obj = bpy.data.objects.new(mesh.name, mesh)
    scene.collection.objects.link(obj)
    material = bpy.data.materials.new("Fixed_albedo_only_M_R_vary")
    material.use_nodes = True
    obj.data.materials.append(material)
    nodes = material.node_tree.nodes
    nodes.clear()
    output_node = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    for key in ("IOR", "Specular IOR Level", "Coat Weight", "Subsurface Weight", "Sheen Weight"):
        bsdf.inputs[key].default_value = protocol["shader"][key]
    texture = nodes.new("ShaderNodeTexImage")
    texture.interpolation = "Closest"
    texture.extension = "EXTEND"
    material.node_tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    material.node_tree.links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])
    if bsdf.inputs["Normal"].is_linked:
        raise ValueError("only actual fixed mesh normals may be used")
    write(output / "scene_audit.json", {"geometry_type": protocol["geometry_type"],
        "mesh_sha256": sha(plan["mesh"]), "camera": protocol["camera"], "lights": protocol["lights"],
        "protocol_sha256": plan["protocol_sha256"], "vertices": len(mesh.vertices), "faces": len(mesh.polygons),
        "normal_input_linked": False, "shader": protocol["shader"],
        "devices": [{"name": device.name, "type": device.type} for device in devices],
        "render_engine": scene.render.engine, "samples": scene.cycles.samples, "seed": scene.cycles.seed})
    return scene, material, texture, bsdf, output_node


def render(plan, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if sha(plan["mesh"]) != plan["mesh_sha256"]:
        raise ValueError("scene geometry changed")
    scene, material, texture, bsdf, output_node = setup(plan, output)
    first = plan["rows"][0]
    texture.image = bpy.data.images.load(first["maps"]["albedo"]["path"], check_existing=True)
    texture.image.colorspace_settings.name = "sRGB"
    # Emission is ONLY an independent pixel-to-UV diagnostic, never training RGB.
    emission = material.node_tree.nodes.new("ShaderNodeEmission")
    material.node_tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs[0], output_node.inputs["Surface"])
    scene.render.filepath = str(output / "uv_diagnostic.exr")
    bpy.ops.render.render(write_still=True)
    material.node_tree.nodes.remove(emission)
    material.node_tree.links.new(bsdf.outputs[0], output_node.inputs["Surface"])
    texture.image = None
    material.node_tree.links.remove(bsdf.inputs["Base Color"].links[0])
    bsdf.inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1)
    bsdf.inputs["Metallic"].default_value = 0
    bsdf.inputs["Roughness"].default_value = 0.5
    scene.use_nodes = True
    bpy.context.view_layer.use_pass_normal = True
    bpy.context.view_layer.use_pass_z = True
    scene.node_tree.nodes.clear()
    layers = scene.node_tree.nodes.new("CompositorNodeRLayers")
    composite = scene.node_tree.nodes.new("CompositorNodeComposite")
    scene.node_tree.links.new(layers.outputs["Image"], composite.inputs[0])
    for name in ("Normal", "Depth"):
        node = scene.node_tree.nodes.new("CompositorNodeOutputFile")
        node.base_path = str(output)
        node.file_slots[0].path = name.lower() + "_"
        node.format.file_format = "OPEN_EXR"
        node.format.color_depth = "32"
        node.format.color_mode = "RGB"
        scene.node_tree.links.new(layers.outputs[name], node.inputs[0])
    scene.render.filepath = str(output / "clay_diagnostic.exr")
    bpy.ops.render.render(write_still=True)
    scene.use_nodes = False
    material.node_tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    texture.image = bpy.data.images.load(first["maps"]["albedo"]["path"], check_existing=True)
    texture.image.colorspace_settings.name = "sRGB"
    texture.image.pack()
    bsdf.inputs["Metallic"].default_value = first["metallic"]
    bsdf.inputs["Roughness"].default_value = first["roughness"]
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "fixed_scene.blend"))
    records = []
    for row in plan["rows"]:
        albedo = Path(row["maps"]["albedo"]["path"])
        if sha(albedo) != row["maps"]["albedo"]["sha256"]:
            raise ValueError("frozen Albedo changed")
        texture.image = bpy.data.images.load(str(albedo), check_existing=True)
        texture.image.colorspace_settings.name = "sRGB"
        bsdf.inputs["Metallic"].default_value = row["metallic"]
        bsdf.inputs["Roughness"].default_value = row["roughness"]
        file = output / (row["id"] + ".exr")
        scene.render.filepath = str(file)
        bpy.ops.render.render(write_still=True)
        records.append({"id": row["id"], "path": str(file), "sha256": sha(file),
                        "albedo_sha256": sha(albedo), "metallic": row["metallic"], "roughness": row["roughness"],
                        "mesh_sha256": plan["mesh_sha256"], "protocol_sha256": plan["protocol_sha256"]})
        write(output / "renders.json", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    render(json.loads(args.plan.read_text()), args.output)
