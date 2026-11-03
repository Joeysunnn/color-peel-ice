"""Bounded single-photo Qwen initializer experiment on the pinned SD2.1 base."""

from pathlib import Path
from types import SimpleNamespace

from experiments.sd15_token_local_kv_v1.contract import read_json
from experiments.sd21_wood_branches_v1.contract import MODEL_DIR
from experiments.subject_multicolor_multimaterial_b.training import build_config as subject_config
from experiments.mailbox_color_sd15_v1.training import build_config as color_config
from experiments.texture_material_disentangled_v1.train import build_config as natural_config

STUDY = 'sd21_qwen_initializers_v1'
BRANCHES = ('subject', 'color', 'material', 'texture')
METHODS = ('lora_r64',)
TOKENS = dict(zip(BRANCHES, ('<S*>', '<C*>', '<M*>', '<T*>')))
STEPS = {'subject': (500, 1000), 'color': (100, 150), 'material': (500, 1000), 'texture': (500, 1000)}
COUNTS = {'subject': 11, 'color': 9, 'material': 8, 'texture': 16}
GPUS = dict(zip(BRANCHES, range(4)))
CONCEPT_IDS = {branch: 'mailbox_' + branch for branch in BRANCHES}
SOURCE_SHA = 'ba9c6624d9998cdfcd16308b0cf8fd8f3778909d8655efe2ba9cebba370483d0'


def build_config(experiment, branch, method='lora_r64', model_dir=MODEL_DIR, smoke=False):
    if branch not in BRANCHES or method not in METHODS or Path(model_dir) != MODEL_DIR or type(smoke) is not bool:
        raise ValueError('require the declared SD2.1 four-branch LoRA64 recipe')
    experiment = Path(experiment).resolve()
    audit = read_json(experiment / 'source_audit.json')
    dataset = experiment / 'dataset' / branch
    if branch == 'subject':
        config = subject_config(method, dataset)
    elif branch == 'color':
        config = color_config(method, dataset)
    else:
        template = {'method': method, 'branch': branch, 'token': TOKENS[branch],
                    'concept_id': 'polished_metal' if branch == 'material' else 'vertical_stripes'}
        config = natural_config(template, experiment / 'dataset', 'pilot')
        config.pop('texture_material')
    config['status'] = 'single_image_qwen_initializers'
    config['run'] = {'study': STUDY, 'variant': branch + ('_smoke2' if smoke else '_fresh'), 'seed': 42}
    config['environment']['CUDA_VISIBLE_DEVICES'] = str(GPUS[branch])
    config['data_manifest'] = str(dataset / 'training_assets_manifest.jsonl')
    config['args'].update(pretrained_model_name_or_path=str(model_dir), concepts_list=str(dataset / 'concepts.json'),
        modifier_token=TOKENS[branch], initializer_token=audit['initializers']['branches'][branch]['initializer'],
        initializer_manifest=audit['initializer_manifest'], max_train_steps=2 if smoke else STEPS[branch][-1],
        checkpoint_steps=[2] if smoke else list(STEPS[branch]), gradient_checkpointing=True,
        concept_isolation_weight=0.0, adam_weight_decay=0.0)
    config['data_lineage'] = {'original_image_sha256': SOURCE_SHA, 'dataset': str(dataset),
        'rows': COUNTS[branch], 'old_weights_reused': False, 'initializer_manifest': audit['initializer_manifest'],
        'material_caption_change': 'made of <M*> -> with a <M*> surface',
        'texture_source': 'Mailbox albedo pattern, not historical wood-spoon texture'}
    return config


def recipe(experiment, branch):
    from .data import verify_dataset
    return SimpleNamespace(BRANCHES=(branch,), METHODS=METHODS, STEPS=STEPS[branch], MODEL_DIR=MODEL_DIR,
        TOKENS=TOKENS, CONCEPT_IDS=CONCEPT_IDS,
        verify_dataset=lambda selected: verify_dataset(Path(experiment) / 'dataset', selected),
        build_config=lambda selected, method, model_dir=MODEL_DIR, smoke=False:
            build_config(experiment, selected, method, model_dir, smoke))
