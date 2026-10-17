"""Uniform Material shader and input guards need no Blender installation."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.material_uniform_albedo_v1 import protocol, render


class Nodes(list):
    def new(self, kind):
        inputs = {name: SimpleNamespace(default_value=None) for name in (
            "Base Color", "Roughness", "Metallic", "IOR", "Specular IOR Level",
            "Coat Weight", "Transmission Weight", "Alpha", "Surface")}
        node = SimpleNamespace(kind=kind, inputs=inputs, outputs={"BSDF": object()})
        self.append(node)
        return node


class UniformMaterialRendererTests(unittest.TestCase):
    def test_shader_has_only_constant_principled_and_output(self):
        frozen = protocol.read_protocol()
        for row in protocol.records(frozen):
            with self.subTest(material=row["material_id"], color=row["color"]):
                nodes, links = Nodes(), []
                shader = SimpleNamespace(node_tree=SimpleNamespace(
                    nodes=nodes, links=SimpleNamespace(new=lambda a, b: links.append((a, b)))))
                bpy = SimpleNamespace(data=SimpleNamespace(materials=SimpleNamespace(new=lambda name: shader)))
                obj = SimpleNamespace(data=SimpleNamespace(materials=["old material"]))
                evidence = render.set_material(obj, row, frozen, bpy)
                self.assertEqual([node.kind for node in nodes],
                                 ["ShaderNodeOutputMaterial", "ShaderNodeBsdfPrincipled"])
                self.assertEqual(len(links), 1)
                bsdf = nodes[1]
                self.assertIs(links[0][0], bsdf.outputs["BSDF"])
                self.assertIs(links[0][1], nodes[0].inputs["Surface"])
                expected_color = [0.5 * x for x in frozen["dataset"]["colors_linear_rgb"][row["color"]]] + [1.0]
                self.assertEqual(bsdf.inputs["Base Color"].default_value, expected_color)
                for name, key in (("Roughness", "roughness"), ("Metallic", "metallic")):
                    self.assertEqual(bsdf.inputs[name].default_value, frozen["materials"][row["material_id"]][key])
                self.assertEqual(evidence["texture_image_nodes"], 0)
                self.assertFalse(evidence["normal_bump_displacement"])
                self.assertEqual(obj.data.materials, [shader])

    def test_changed_material_or_records_rejected_before_blender(self):
        frozen = protocol.read_protocol()
        rows = protocol.records(frozen)
        plan = {"status": "complete", "protocol_sha256": protocol.protocol_sha256(frozen), "records": rows}
        self.assertEqual(render.validate_plan(plan, frozen), rows)
        for field, value in (("roughness", 0.99), ("material_id", "not_a_material"),
                             ("seed", 999), ("albedo_level", 0.75)):
            changed = copy.deepcopy(plan)
            changed["records"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Plan records differ"):
                render.validate_plan(changed, frozen)
        changed = copy.deepcopy(plan)
        changed["records"].append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "Plan records differ"):
            render.validate_plan(changed, frozen)

    def test_unsafe_identifier_is_rejected_before_blender(self):
        frozen = protocol.read_protocol()
        for identifier in ("../old", "/absolute", "..", "a/b", "a\\b"):
            rows = protocol.records(frozen)
            rows[0]["id"] = identifier
            plan = {"status": "complete", "protocol_sha256": protocol.protocol_sha256(frozen), "records": rows}
            with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, "Unsafe render ID"):
                render.validate_plan(plan, frozen)

    def test_existing_output_is_preserved_before_plan_read(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "old_result"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                render.main(["--plan", "missing.json", "--output", directory, "--gpu", "3"])
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
