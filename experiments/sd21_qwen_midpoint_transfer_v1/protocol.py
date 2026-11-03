"""Requested learned-only transfer probes for saved Qwen midpoint weights."""

from pathlib import Path

from experiments.sd21_qwen_initializers_v1.inference import BRANCHES, TOKENS, SAMPLING

STUDY = 'sd21_qwen_midpoint_transfer_v1'
SOURCE_GIT = {'branch': 'codex/sd21-qwen-initializers-v1',
              'commit': 'cb169c6f1b4b8631c5dfae373145e217abf26ae3'}
SOURCE = Path('/home/r12user5/Documents/Jiawei/colorpeel-runs/sd21_qwen_initializers_v1/'
              '20261103-174000__mailbox_four__cb169c6__42/experiment')
STEPS = {'subject': 500, 'color': 100, 'material': 500, 'texture': 500}
SEEDS = tuple(range(30, 41))
GPUS = {'subject_only': 0, 'scmt': 0, 'material_only': 1, 'texture_only': 2, 'color_only': 3}
SCENES = {'plain': '', 'street': 'in the street', 'snow': 'in the snow', 'flowers': 'surrounded by flowers'}
COLOR_PROMPTS = {
    'bowl': 'a <C*> bowl on the table',
    'vase': 'a <C*> vase on the shelf',
    'parrot': 'a <C*> parrot perched on a tree',
    'sofa': 'a <C*> sofa in living room',
    'rose': 'a <C*> rose blooming in a wooden pot',
}


def plan(manifest):
    entries = manifest['branches']
    if (manifest['subject_category'] != 'mailbox' or set(entries) != set(BRANCHES)
            or any(entries[b]['token'] != TOKENS[b] for b in BRANCHES)):
        raise ValueError('this fixed probe requires the audited Mailbox four-branch manifest')
    prompts = {
        'subject_only': {case: 'a photo of <S*> mailbox ' + suffix for case, suffix in {
            'white': 'in white color', 'red': 'in red color', 'snow': 'in the snow',
            'street': 'in the street', 'flowers': 'surrounded by flowers'}.items()},
        'material_only': {obj: f'a photo of a {obj} with a <M*> surface'
                          for obj in ('chair', 'vase', 'sofa', 'mug')},
        'texture_only': {'hair': 'a photo of hair with <T*> texture', **{
            obj: f'a photo of a {obj} with <T*> texture' for obj in ('vase', 'sofa', 'mug')}},
        'color_only': COLOR_PROMPTS,
        'scmt': {case: ('a photo of <S*> mailbox in <C*> color with a <M*> surface '
                        'with <T*> texture ' + suffix).strip() for case, suffix in SCENES.items()},
    }
    rows = []
    for mode, cases in prompts.items():
        branches = BRANCHES if mode == 'scmt' else (mode.removesuffix('_only'),)
        for case, prompt in cases.items():
            for seed in SEEDS:
                identity = f'mid__{mode}__learned__{case}__seed{seed}'
                rows.append({'id': identity, 'phase': 'mid', 'mode': mode, 'condition': 'learned',
                    'case': case, 'method': 'lora_r64', 'seed': seed, 'prompt': prompt,
                    'negative_prompt': '', 'guidance_scale': 6.0 if mode == 'color_only' else 3.5,
                    'branch_steps': {branch: STEPS[branch] for branch in branches},
                    'target_initializers': {branch: entries[branch]['initializer'] for branch in branches},
                    'image_path': f'images/{mode}/{case}/seed{seed}.png'})
    return rows
