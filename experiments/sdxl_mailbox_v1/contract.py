"""Mailbox-only SDXL data, training dose and independent transfer matrix."""

METHODS = ('fullrank', 'lora_r64')
BRANCHES = ('subject', 'color', 'material', 'texture')
SEEDS = tuple(range(30, 41))
RESOLUTION, LATENT_SIZE = 1024, 128
TRAINING_SEED = 42
EMBEDDING_LR = 1e-5
KV_LR = {'fullrank': 1e-5, 'lora_r64': 5e-5}
TRAINING = {
    'subject': {'token': '<S*>', 'initializer': 'mailbox', 'images': 27,
                'steps': (400, 600, 800, 1000), 'max_steps': 1000, 'cfg': 3.5},
    'color': {'token': '<C*>', 'initializer': 'orange', 'images': 9,
              'steps': (100, 150), 'max_steps': 150, 'cfg': 6.0},
    'material': {'token': '<M*>', 'initializer': 'metal', 'images': 8,
                 'steps': (400, 600, 800, 1000), 'max_steps': 1000, 'cfg': 3.5},
    'texture': {'token': '<T*>', 'initializer': 'texture', 'images': 16,
                'steps': (400, 600, 800, 1000), 'max_steps': 1000, 'cfg': 3.5},
}
COLOR_PROMPTS = (
    'a <C*> bowl on the table',
    'a <C*> bowling ball in a bowling alley',
    'a <C*> plate on the table',
    'a <C*> vase on the shelf',
    'a women wearing <C*> pants',
    'a <C*> teddy-bear in Time Square',
    'a <C*> snooker ball on the table',
    'a <C*> parrot perched on a tree',
    'a <C*> sofa in living room',
    'a <C*> rose blooming in a wooden pot',
)


def transfer_cases(branch, steps, smoke=False):
    """Return a method-independent matrix; native/literal prompts have zero KV gates."""
    spec = TRAINING[branch]
    steps = (steps,) if type(steps) is int else tuple(steps)
    if (not steps or len(set(steps)) != len(steps)
            or any(type(step) is not int or step < 1 for step in steps)
            or (not smoke and any(step not in spec['steps'] for step in steps))):
        raise ValueError('require declared paired checkpoints')
    if branch == 'subject':
        prompts = [('learned_subject', f'a photo of <S*> mailbox with a {material} surface')
                   for material in ('metallic', 'matte', 'glossy plastic', 'ceramic', 'wood')]
        prompts += [('learned_subject', f'a photo of <S*> mailbox {condition}') for condition in
                    ('in the snow', 'surrounded by flowers', 'in white color', 'in orange color')]
    elif branch == 'color':
        prompts = [('learned_color', prompt) for prompt in COLOR_PROMPTS]
    else:
        literal = 'painted metal' if branch == 'material' else 'fine speckled'
        template = ('a photo of a {object} made of {attribute}' if branch == 'material'
                    else 'a photo of a {object} with {attribute} texture')
        prompts = []
        for obj in ('chair', 'mug', 'vase', 'sofa'):
            prompts.extend((('base', f'a photo of a {obj}'),
                            ('literal', template.format(object=obj, attribute=literal)),
                            ('learned_' + branch, template.format(object=obj, attribute=spec['token']))))
    if smoke:
        prompts = [next(row for row in prompts if row[0].startswith('learned_'))]
    return [{'id': f'{branch}__step{step}__prompt{index:02d}__seed{seed}',
             'branch': branch, 'step': step, 'seed': seed, 'condition': condition, 'prompt': prompt}
            for step in steps for index, (condition, prompt) in enumerate(prompts)
            for seed in (SEEDS[:1] if smoke else SEEDS)]
