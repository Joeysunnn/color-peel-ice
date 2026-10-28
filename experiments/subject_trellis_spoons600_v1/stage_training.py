"""Byte-copy the frozen spoon blue/green72 sources for the authorized600-step dose."""

from pathlib import Path

from experiments.subject_trellis_dataset_v1 import stage_training as shared
from experiments.subject_trellis_spoons_v1 import stage_training as previous

REQUEST = ('那就用两把勺子的blue+green72张数据，original和rank64各训练600步，'
    '保存200/400/600步并做相同的材质迁移测试。每个实验放同一文件夹，'
    '严格本地修改提交推送、服务器拉取运行，旧结果不要动，启动后不用监控。')
SOURCES = {subject: {**spec, 'request': REQUEST, 'max_train_steps': 600,
    'checkpoint_steps': [200, 400, 600]} for subject, spec in previous.SOURCES.items()}


def stage(candidates, output, blue_green_only=True):
    spec = previous.source_for(candidates)
    if not blue_green_only:
        raise ValueError('this experiment accepts only the blue/green72 subset')
    return shared.stage(candidates, output, True, source_spec=SOURCES[spec['subject_id']])


def verify_dataset(dataset):
    dataset = Path(dataset).resolve()
    subject = shared.read(dataset / 'dataset_audit.json').get('subject_id')
    if subject not in SOURCES:
        raise ValueError('unknown spoon subject')
    audit = shared.verify_dataset(dataset, source_spec=SOURCES[subject])
    approval = shared.read(dataset / 'approval.json')
    if (audit['rows'] != 72 or set(audit['selected_colors']) != {'blue', 'green'}
            or audit['sampling']['updates'] != 600 or approval.get('max_train_steps') != 600
            or approval.get('checkpoint_steps') != [200, 400, 600]):
        raise ValueError('require exactly72 source images and the authorized600-step checkpoint plan')
    return audit
