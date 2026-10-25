"""Strength-only identity ablation through the unmodified official caller."""


def rows():
    return [{"id": f"seed42__strength{value:.2f}", "controls": {}, "reference": "self",
             "target_size": 1024, "seed": 42, "strength": value,
             "label": f"s42 | no MLP | strength={value:.2f}"}
            for value in (1.0, .75, .5, .25)]


class StrengthOverride:
    """Override only the official caller's hard-coded strength, without patching it."""

    def __init__(self, adapter, value):
        self.adapter = adapter
        self.value = value

    def generate_parametric_edits(self, *args, **kwargs):
        if kwargs["strength"] != 1.0:
            raise ValueError("official strength default changed")
        kwargs["strength"] = self.value
        return self.adapter.generate_parametric_edits(*args, **kwargs)


def write_review(output, generated, sheet):
    from PIL import Image

    with Image.open(output / "source/review_mask.png") as image:
        x0, y0, x1, y1 = image.convert("L").getbbox()
    crop = (max(0, x0 - 24), max(0, y0 - 24), min(1024, x1 + 24), min(1024, y1 + 24))
    sources = [("Target = self reference", "source/target.png"),
               ("Official gray-subject init", "source/initialization.png"),
               ("Frozen mailbox Mask", "source/mask.png"), ("Fresh1024 DPT", "source/depth.png")]
    panels = [{"raw_image": "source/target.png", "label": "Original input"},
              {"raw_image": "source/initialization.png", "label": "Actual grayscale-subject init"}] + generated
    sheet(output, panels, sources, "strength_full.png",
          title="Strength-only identity test | self reference / seed42 / nominal30 steps | NO training")
    sheet(output, panels, sources, "strength_subject.png", crop=crop,
          title="Strength-only | original / actual init / 1 / .75 / .5 / .25 | Albedo unverified")
