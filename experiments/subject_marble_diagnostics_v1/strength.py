"""Strength-only identity ablation through the unmodified official caller."""


def rows():
    return [{"id": f"seed42__strength{value:.2f}", "controls": {}, "reference": "self",
             "target_size": 1024, "seed": 42, "strength": value,
             "label": f"s42 | no MLP | strength={value:.2f}"}
            for value in (1.0, .75, .5, .25)]


def rgb_rows():
    return [{**row, "id": mode + "__" + row["id"], "initialization": mode,
             "initialization_image": "source/initialization_rgb.png" if mode == "rgb" else "source/initialization.png",
             "label": f"{mode.upper()} | strength={row['strength']:.2f}"}
            for mode in ("gray", "rgb") for row in rows()]


class StrengthOverride:
    """Override authorized caller inputs without patching official code or weights."""

    def __init__(self, adapter, value, init_image=None):
        self.adapter = adapter
        self.value = value
        self.init_image = init_image

    def generate_parametric_edits(self, *args, **kwargs):
        if kwargs["strength"] != 1.0:
            raise ValueError("official strength default changed")
        kwargs["strength"] = self.value
        if self.init_image is not None:
            kwargs["image"] = self.init_image.copy()
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


def write_rgb_review(output, generated, sheet):
    from PIL import Image

    with Image.open(output / "source/review_mask.png") as image:
        x0, y0, x1, y1 = image.convert("L").getbbox()
    crop = (max(0, x0 - 24), max(0, y0 - 24), min(1024, x1 + 24), min(1024, y1 + 24))
    sources = [("Original RGB init = reference", "source/initialization_rgb.png"),
               ("Official gray-subject init", "source/initialization.png"),
               ("Same mailbox Mask", "source/mask.png"), ("Fresh1024 DPT", "source/depth.png")]
    sheet(output, generated, sources, "rgb_init_full.png", columns=4,
          title="Init identity ablation | rows:GRAY / RGB | columns:1/.75/.5/.25 | seed42 / no MLP")
    sheet(output, generated, sources, "rgb_init_comparison_subject.png", columns=4, crop=crop,
          title="GRAY vs RGB init | columns:1/.75/.5/.25 | seed42 / no MLP | Albedo unverified")
    for mode in ("gray", "rgb"):
        selected = [row for row in generated if row["initialization"] == mode]
        panels = [{"raw_image": "source/target.png", "label": "Original mailbox input"},
                  {"raw_image": selected[0]["initialization_image"], "label": mode.upper() + " actual initialization"}] + selected
        sheet(output, panels, sources, f"{mode}_init_subject.png", crop=crop,
              title=f"{mode.upper()} init | original / actual init / 1/.75/.5/.25 | Albedo unverified")


def write_rgb_mr_review(output, generated, sheet):
    from PIL import Image

    with Image.open(output / "source/review_mask.png") as image:
        x0, y0, x1, y1 = image.convert("L").getbbox()
    crop = (max(0, x0 - 24), max(0, y0 - 24), min(1024, x1 + 24), min(1024, y1 + 24))
    sources = [("Mailbox original RGB", "source/target.png"),
               ("Actual RGB init = self ref", "source/initialization_rgb.png"),
               ("Frozen mailbox Mask", "source/mask.png"), ("Fresh1024 DPT", "source/depth.png")]
    sheet(output, generated, sources, "rgb_mr_full.png",
          title="RGB init / strength.25 / seed42 | rows M=-20,0,+20 | columns R=-1,0,+1 | relative controls")
    sheet(output, generated, sources, "rgb_mr_subject.png", crop=crop,
          title="RGB init / strength.25 | M rows / R columns | visible content + reflectance | Albedo unverified")
