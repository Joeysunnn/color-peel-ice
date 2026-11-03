"""Controlled Material labels must remain distinct from their initialization prior."""

import copy
import itertools
import unittest

from src.methods.colorpeel_ice.material_initialization import (
    CONFIDENCES, INITIALIZERS, LABELS, POLICY_VERSION, SCHEMA,
    select_material_initializer, validate_material,
)


def observation(metallicity="dielectric", roughness="low", m_conf="high", r_conf="high"):
    return {"metallicity": {"label": metallicity, "confidence": m_conf,
                            "evidence": "Visible surface reflection evidence"},
            "roughness": {"label": roughness, "confidence": r_conf,
                          "evidence": "Observed highlight sharpness"}}


class ControlledMaterialPolicyTests(unittest.TestCase):
    def test_visible_reflection_extremes_and_generic_middle_have_explicit_decisions(self):
        examples = [
            (observation(), "glossy", False),
            (observation(roughness="high"), "matte", False),
            (observation(roughness="medium"), "material", True),
            (observation(metallicity="uncertain", roughness="low", m_conf="low", r_conf="medium"), "glossy", False),
            (observation(metallicity="uncertain", roughness="high", m_conf="low", r_conf="medium"), "matte", False),
            (observation(metallicity="uncertain", roughness="uncertain", m_conf="low", r_conf="low"), "material", True),
        ]
        for value, word, fallback in examples:
            with self.subTest(value=value):
                result = select_material_initializer(value)
                self.assertEqual(result["initializer"], word)
                self.assertEqual(result["fallback"], fallback)
                self.assertEqual(result["policy_version"], POLICY_VERSION)
                self.assertTrue(result["selection_reason"])
        self.assertEqual(SCHEMA, "colorpeel_vlm_initializers/v2")

    def test_priority_preserves_two_axes_without_claiming_word_encodes_both(self):
        value = observation(metallicity="metallic", roughness="high")
        before = copy.deepcopy(value)
        result = select_material_initializer(value)
        self.assertEqual(result["initializer"], "metallic")
        self.assertEqual(result["selection_reason"], "high_confidence_metallic")
        self.assertEqual(value, before)
        self.assertEqual(value["roughness"]["label"], "high")
        self.assertEqual(select_material_initializer(observation("metallic", "high", "medium"))["initializer"], "matte")

    def test_mixed_and_low_confidence_do_not_force_material_identity(self):
        for roughness, confidence in itertools.product(LABELS["roughness"], CONFIDENCES):
            with self.subTest(roughness=roughness, confidence=confidence):
                result = select_material_initializer(observation("mixed", roughness, "high", confidence))
                self.assertEqual(result["initializer"], "material")
                self.assertEqual(result["selection_reason"], "mixed_metallicity")
                self.assertTrue(result["fallback"])
        for label in ("low", "high"):
            self.assertEqual(select_material_initializer(observation(roughness=label, r_conf="low"))["initializer"], "material")

    def test_all_valid_label_combinations_stay_in_pool_and_do_not_modify_evidence(self):
        for m_label, r_label, m_conf, r_conf in itertools.product(
                LABELS["metallicity"], LABELS["roughness"], CONFIDENCES, CONFIDENCES):
            value = observation(m_label, r_label, m_conf, r_conf)
            before = copy.deepcopy(value)
            result = select_material_initializer(value)
            self.assertIn(result["initializer"], INITIALIZERS)
            self.assertEqual(set(result), {"policy_version", "initializer", "selection_reason", "fallback"})
            self.assertIs(type(result["fallback"]), bool)
            self.assertEqual(value, before)

    def test_free_words_missing_fields_extra_fields_and_invalid_types_are_rejected(self):
        invalid = [None, [], "painted", {}, {**observation(), "candidates": ["painted"]}]
        for axis in LABELS:
            changed = observation()
            changed.pop(axis)
            invalid.append(changed)
            for replacement in (None, [], "glossy", {}, {**observation()[axis], "number": 0.3}):
                changed = observation()
                changed[axis] = replacement
                invalid.append(changed)
            for field, values in (("label", ("painted", "smooth", "glossy", "Low", 0.3, [], None)),
                                  ("confidence", ("unknown", 0.9, [], None)),
                                  ("evidence", ("", "  ", 7, [], None))):
                for replacement in values:
                    changed = observation()
                    changed[axis][field] = replacement
                    invalid.append(changed)
            for field in ("label", "confidence", "evidence"):
                changed = observation()
                changed[axis].pop(field)
                invalid.append(changed)
        for index, value in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_material(value)
            with self.subTest(index=index), self.assertRaises(ValueError):
                select_material_initializer(value)


if __name__ == "__main__":
    unittest.main()
