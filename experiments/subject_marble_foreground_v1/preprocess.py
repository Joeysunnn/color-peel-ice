"""Remove original-scene pixels before every MARBLE image-conditioned branch.

Target coordinates remain unchanged. The separate padded reference increases
the subject's CLIP occupancy, so this is a joint foreground/reference adaptation,
not a background-only single-factor comparison.
"""

import math

from PIL import Image

NEUTRAL = (127, 127, 127)


def _mask(mask, size):
    if mask.mode != "L" or mask.size != size:
        raise ValueError("require an aligned L-mode binary mask")
    if {value for _, value in mask.getcolors(256)} != {0, 255}:
        raise ValueError("require a nonempty binary mask with foreground and background")


def prepare_inputs(original, mask):
    """Return isolated512 originals and aligned1024 generation conditions."""
    if original.mode != "RGB" or original.size != (512, 512):
        raise ValueError("require the original512 RGB image")
    _mask(mask, original.size)
    neutral = Image.new("RGB", original.size, NEUTRAL)
    original_neutral = Image.composite(original, neutral, mask)
    cutout = Image.composite(original, Image.new("RGB", original.size), mask).convert("RGBA")
    cutout.putalpha(mask)

    mask1024 = mask.resize((1024, 1024), Image.Resampling.NEAREST)
    # Mask before interpolation to prevent scene pixels bleeding into the edge.
    # Restore the binary-mask exterior after interpolation as well.
    target = Image.composite(
        original_neutral.resize((1024, 1024), Image.Resampling.BICUBIC),
        Image.new("RGB", (1024, 1024), NEUTRAL), mask1024)
    left, top, right, bottom = mask1024.getbbox()
    margin = math.ceil(max(right - left, bottom - top) * .1)
    box = (max(0, left - margin), max(0, top - margin),
           min(1024, right + margin), min(1024, bottom + margin))
    reference = target.crop(box)
    scale = min(1024 / reference.width, 1024 / reference.height)
    size = (max(1, round(reference.width * scale)), max(1, round(reference.height * scale)))
    reference = reference.resize(size, Image.Resampling.BICUBIC)
    padded = Image.new("RGB", (1024, 1024), NEUTRAL)
    padded.paste(reference, ((1024 - size[0]) // 2, (1024 - size[1]) // 2))
    return {"target": target, "reference": padded, "mask": mask1024, "cutout": cutout,
            "original_neutral": original_neutral, "reference_box": box}


def mask_depth(depth, mask):
    """Return RGB1024 depth, with only the original foreground values retained."""
    _mask(mask, (1024, 1024))
    if depth.mode not in ("L", "RGB") or depth.size != (1024, 1024):
        raise ValueError("require aligned1024 L/RGB estimated depth")
    return Image.composite(depth.convert("RGB"), Image.new("RGB", depth.size), mask)
