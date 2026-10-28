"""Stage only the two frozen spoon sources, preserving their independent identity."""

from pathlib import Path

from experiments.subject_trellis_dataset_v1 import stage_training as shared

REQUEST = ('用这两套各180张数据，blue+green72张数据 （两组放在同一文件夹下）分别训练subject token，'
    '沿用之前original和rank64两组设置，各1000步，保存500/750/1000步并做相同的材质迁移测试。'
    '每个实验放在同一个文件夹下，严格本地修改提交推送、服务器拉取运行，旧结果不要动，启动后不用监控。')
SOURCE_ROOT = Path('/var/tmp/colorpeel-r12user5/runs/subject_trellis_spoons_v1/'
    '20261028-013740__assets__2943bf6__42/datasets')
SOURCES = {
    'metal_spoon': {
        'manifest_sha256': 'dae40c7f2a1f385357b8d5628a6a50a1e920bd24848d2e995af573bc9de5c710',
        'concepts_sha256': 'bab170ad03ecb3652202d3104f237e3a9ea891d65d235848dc45adc2d52b0ec6'},
    'wood_spoon': {
        'manifest_sha256': 'b5658a2ec6e124c6fdb7e946635e3e248cd0e4458173fd4c4d44029c73d7b348',
        'concepts_sha256': '58bb0f033a0ed60679d9112f7f299668a836b369a8a9741ccf0aaed8c3d1574e'},
}
for subject_id, spec in SOURCES.items():
    spec.update(subject_id=subject_id, candidates=SOURCE_ROOT / subject_id / 'dataset',
        protocol=Path(__file__).with_name(subject_id + '.json'), request=REQUEST)


def source_for(candidates):
    candidates = Path(candidates).resolve()
    matches = [spec for spec in SOURCES.values() if Path(spec['candidates']).resolve() == candidates]
    if len(matches) != 1:
        raise ValueError('only the frozen complete metal_spoon and wood_spoon sources are authorized')
    return matches[0]


def stage(candidates, output, blue_green_only=False):
    return shared.stage(candidates, output, blue_green_only, source_spec=source_for(candidates))


def verify_dataset(dataset):
    subject = shared.read(Path(dataset) / 'dataset_audit.json').get('subject_id')
    if subject not in SOURCES:
        raise ValueError('unknown or missing staged spoon subject')
    return shared.verify_dataset(dataset, source_spec=SOURCES[subject])
