"""The ten historical Color prompts plus three explicitly requested S+C probes."""

import itertools
import json
from pathlib import Path

from experiments.sd15_token_local_kv_v1.contract import sha256

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/natural_image_subject_color_pilot/configs/d1_mailbox_orange_shared_kv_joint_1000_color_transfer_protocol_v1.json'
METHODS = ('fullrank', 'lora_r64')
STEPS = (400, 600, 800, 1000)
SEEDS = tuple(range(30, 61))
JOINT_PROMPTS = (
    'a photo of <S*> mailbox in <C*> color',
    'a photo of <S*> mailbox in <C*> color in the snow',
    'a photo of <S*> mailbox in <C*> color surrounded by flowers',
)
SAMPLING = {'scheduler': 'PNDM', 'num_inference_steps': 100, 'guidance_scale': 6.0,
            'width': 512, 'height': 512, 'dtype': 'float16', 'safety_checker': True,
            'generator': 'fresh_device_generator_per_image', 'seeds': list(SEEDS)}


def conditions():
    old = json.loads(SOURCE.read_text(encoding='utf8'))['prompts']
    if len(old) != 10 or len({row['prompt'] for row in old}) != 10:
        raise ValueError('require the ten historical Color transfer prompts')
    result = [{'condition': row['color'], 'prompt': row['prompt'], 'arm': 'color_only'} for row in old]
    result.extend({'condition': label, 'prompt': prompt, 'arm': 'subject_color'}
                  for label, prompt in zip(('subject_color', 'subject_color_snow', 'subject_color_flowers'), JOINT_PROMPTS))
    for row in result:
        if (row['prompt'].count('<C*>') != 1
                or row['prompt'].count('<S*>') != int(row['arm'] == 'subject_color')):
            raise ValueError('each probe requires one Color token and only joint probes require one Subject token')
    return result


def prompt_rows(method, step):
    if method not in METHODS or type(step) is not int or step not in STEPS:
        raise ValueError('require a declared method and Color checkpoint step')
    return [{'id': f'{method}__step{step}__{condition["condition"]}__seed{seed}',
             'method': method, 'step': step, **condition, 'seed': seed,
             'image_path': f'images/{condition["condition"]}__seed{seed}.png'}
            for condition, seed in itertools.product(conditions(), SEEDS)]


def source_record():
    return {'path': str(SOURCE), 'sha256': sha256(SOURCE), 'historical_prompt_count': 10,
            'joint_prompts': list(JOINT_PROMPTS)}
