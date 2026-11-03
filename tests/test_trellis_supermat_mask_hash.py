from types import SimpleNamespace

import numpy as np
from PIL import Image, PngImagePlugin
import pytest

from experiments.trellis_supermat_compare_v1.render_compare import mask_pixel_sha256, sha256, verify_pair


def test_metadata_independent_exact_mask_comparison(tmp_path):
    image = Image.fromarray(np.array([[0, 64], [128, 255]], dtype=np.uint8))
    paths = [tmp_path / 'first.png', tmp_path / 'second.png']
    for path, date in zip(paths, ('00:00:01', '00:00:02')):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text('Date', date)
        image.save(path, pnginfo=metadata)
    assert sha256(paths[0]) != sha256(paths[1])
    first = {key: None for key in ('shape', 'color', 'light', 'seed', 'camera', 'lights',
        'uniform_base_color_linear_rgba', 'world_strength', 'object_scale', 'resolution',
        'cycles_samples', 'color_management', 'cycles_device')}
    first.update(mask_sha256=sha256(paths[0]), mask_pixel_sha256=mask_pixel_sha256(paths[0]))
    second = dict(first, mask_sha256=sha256(paths[1]), mask_pixel_sha256=mask_pixel_sha256(paths[1]))
    verify_pair(first, second)
    image.putpixel((0, 0), 1)
    image.save(paths[1])
    second['mask_pixel_sha256'] = mask_pixel_sha256(paths[1])
    with pytest.raises(ValueError, match='mask_pixel_sha256'):
        verify_pair(first, second)


def test_blender_bottom_up_pixels_match_pillow(tmp_path):
    path = tmp_path / 'mask.png'
    Image.fromarray(np.array([[0, 64], [128, 255]], dtype=np.uint8)).save(path)
    with Image.open(path) as image:
        bottom_up = (np.array(image.convert('RGBA'))[::-1].astype(np.float32) / 255).ravel()
    loaded = SimpleNamespace(size=(2, 2), colorspace_settings=SimpleNamespace(name='sRGB'),
                             pixels=SimpleNamespace(foreach_get=lambda values: np.copyto(values, bottom_up)))
    removed = []
    images = SimpleNamespace(load=lambda *args, **kwargs: loaded, remove=removed.append)
    bpy = SimpleNamespace(data=SimpleNamespace(images=images))
    assert mask_pixel_sha256(path, bpy) == mask_pixel_sha256(path)
    assert loaded.colorspace_settings.name == 'Non-Color'
    assert removed == [loaded]
