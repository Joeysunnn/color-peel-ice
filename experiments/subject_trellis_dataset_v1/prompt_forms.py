"""Literal material-template comparison; intentionally retain the matte wording."""

import itertools

from experiments.subject_multicolor_multimaterial_b.inference import METHODS, SAMPLING, STEPS

MATERIALS = ('wood', 'metal', 'matte', 'plastic', 'ceramic')
FORMS = ('adjective', 'entirely', 'surface')
LABELS = tuple(f'{material} / {form}' for material, form in itertools.product(MATERIALS, FORMS))
_TEMPLATES = {'adjective': 'a {adjective} <S*>',
              'entirely': 'a <S*> made entirely of {material}',
              'surface': 'a <S*> with a {material} surface'}


def prompt_rows(step, seeds=None):
    if type(step) is not int or step not in STEPS:
        raise ValueError('expected a declared Subject checkpoint step')
    seeds = SAMPLING['seeds'] if seeds is None else seeds
    if (not isinstance(seeds, (tuple, list)) or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError('expected a nonempty unique tuple/list of nonnegative integer seeds')
    seeds = tuple(seeds)
    return [{'id': f'{method}__{material}__{form}__seed{seed}', 'method': method,
             'material': material, 'form': form, 'condition': f'{material} / {form}',
             'prompt': _TEMPLATES[form].format(material=material,
                adjective='wooden' if material == 'wood' else material),
             'seed': seed, 'step': step,
             'image_path': f'images/{method}/{material}__{form}__seed{seed}.png'}
            for method, material, form, seed in itertools.product(METHODS, MATERIALS, FORMS, seeds)]
