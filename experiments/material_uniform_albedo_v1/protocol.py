"""Frozen uniform-albedo material controls; standard-library only for Blender."""

import hashlib
import itertools
import json
from pathlib import Path


PARENT = Path(__file__).resolve().parents[1] / "texture_material_disentangled_v1/protocol.json"
PARENT_SHA256 = "3563eb6c263e9d4c4b6c57f834a789e3a7df1687cfc3d768fd033024e62cf116"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf8")).hexdigest()


def read_protocol():
    protocol = json.loads(PARENT.read_text(encoding="utf8"))
    if _digest(protocol) != PARENT_SHA256:
        raise ValueError("Frozen parent protocol changed")
    protocol.update(study="material_uniform_albedo_v1", parent_protocol_sha256=PARENT_SHA256,
                    representation={"albedo_mode": "uniform", "albedo_level": 0.5,
                                    "texture_image_nodes": 0, "normal_bump_displacement": False})
    protocol.pop("textures")
    protocol.pop("texture_extraction")
    protocol["materials"] = {name: spec for name, spec in protocol["materials"].items()
                             if spec["role"] == "target"}
    protocol["dataset"].update(expected_train_images=16, expected_per_concept=8)
    protocol["training"].update(steps=1000, checkpoint_steps=[500, 750, 1000])
    return protocol


def protocol_sha256(protocol=None):
    return _digest(read_protocol() if protocol is None else protocol)


def records(protocol=None):
    protocol = read_protocol() if protocol is None else protocol
    dataset = protocol["dataset"]
    contexts = list(itertools.product(dataset["train_shapes"], dataset["train_colors"],
                                      dataset["train_lights"]))
    return [{"id": f"train__material__{material}__{shape}__{color}__{light}",
             "split": "train", "branch": "material", "concept_id": material,
             "material_id": material, "shape": shape, "color": color, "light": light,
             "seed": protocol["seed"] + index, "albedo_mode": "uniform", "albedo_level": 0.5}
            for material in protocol["materials"]
            for index, (shape, color, light) in enumerate(contexts)]


def caption(row, protocol=None):
    protocol = protocol or read_protocol()
    light = protocol["dataset"]["light_phrases"][row["light"]]
    return f'a photo of a {row["color"]} {row["shape"]} made of <M*> under {light}'
