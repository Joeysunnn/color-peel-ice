"""Package a completed extraction/render comparison; never train or alter source images."""

import argparse
from collections import Counter
import html
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

from experiments.trellis_supermat_compare_v1.render_compare import sha256, verify_pair


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sheet(items, path, columns, crop=None):
    tile, header = 320, 60
    canvas = Image.new('RGB', (columns * tile, ((len(items) + columns - 1) // columns) * (tile + header)), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 15)
    except OSError:
        font = ImageFont.load_default()
    for index, (image_path, title) in enumerate(items):
        x, y = index % columns * tile, index // columns * (tile + header)
        draw.multiline_text((x + 6, y + 5), title, fill='black', font=font, spacing=3)
        with Image.open(image_path) as opened:
            image = opened.convert('RGB')
            if crop is not None:
                if image.size != (512, 512):
                    raise ValueError('ARM panels must retain their original512 canvas')
                image = image.crop(crop)
            image = ImageOps.contain(image, (tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + (tile - image.width) // 2, y + header + (tile - image.height) // 2))
    canvas.save(path)


def build(run, output, render_dir=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    archive = output.with_name(output.name + '.zip')
    if output.exists() or archive.exists() or output == run or run.is_relative_to(output):
        raise ValueError('Require a fresh preview directory and ZIP separate from the run inputs')
    render_dir = Path(render_dir).resolve() if render_dir else run / 'renders'
    maps, renders = read(run / 'maps/report.json'), read(render_dir / 'manifest.json')
    records = renders['records']
    if (maps['status'] != 'ready_for_review' or renders['status'] != 'complete'
            or renders['expected_images'] != 16 or len(records) != 16
            or Counter(row['method'] for row in records) != {'supermat': 8, 'trellis': 8}):
        raise ValueError('Require successful map extraction and all16 completed renders')
    contexts = set()
    for first, second in zip(records[::2], records[1::2]):
        verify_pair(first, second)
        if (first['method'], second['method']) != ('supermat', 'trellis'):
            raise ValueError('Require SuperMat/TRELLIS adjacent pairs')
        contexts.add((first['shape'], first['color'], first['light']))
    if contexts != set(itertools.product(('sphere', 'cube'), ('red', 'blue'), ('soft_front', 'side_directional'))):
        raise ValueError('Render contexts differ from Material8')
    files = {'original_mailbox.jpg': run / 'input/original_mailbox.jpg',
             'source_cutout.png': run / 'input/inputs/mailbox.png',
             'source_mask.png': run / 'input/object_mask.png', 'reviewed_roi.png': run / 'input/reviewed_roi.png',
             'common_roi.png': run / 'maps/common_roi.png', 'fitted_mask.png': run / 'trellis/review/fitted_mask.png',
             'trellis_pbr.png': run / 'trellis/review/pbr.png', 'trellis_alpha.png': run / 'maps/trellis_alpha.png',
             'conditioning_comparison.png': run / 'maps/conditioning_comparison.png'}
    for method in ('supermat', 'trellis'):
        for channel in ('albedo', 'roughness', 'metallic'):
            native = 'base_color' if method == 'trellis' and channel == 'albedo' else channel
            folder = run / ('supermat/mailbox' if method == 'supermat' else 'trellis/review')
            files[f'{method}_{channel}.png'] = folder / (native + '.png')
    files.update({'panels/' + path.name: path for path in sorted((run / 'maps/panels').glob('*.png'))})
    required = {'supermat_albedo', 'supermat_roughness', 'supermat_metallic', 'trellis_base_color',
                'trellis_roughness', 'trellis_metallic', 'reviewed_roi_overlay', 'common_roi_overlay', 'trellis_roi_overlay'}
    if any(f'panels/{name}.png' not in files for name in required):
        raise ValueError('Required original-canvas map panels are missing')
    for row in records:
        identifier = row['id']
        if not identifier or any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for char in identifier):
            raise ValueError('Unsafe render image identifier')
        files[f'renders/{identifier}.png'] = Path(row['image'])
    expected = {**maps['source_locks'], str(run / 'maps/common_roi.png'): maps['common_roi_sha256'],
                **{row['image']: row['image_sha256'] for row in records}}
    receipts = {name: {'source': str(path), 'sha256': sha256(path)} for name, path in files.items()}
    if any(str(files[name]) in expected and value['sha256'] != expected[str(files[name])] for name, value in receipts.items()):
        raise ValueError('Source image hash changed')
    with Image.open(files['source_mask.png']) as source, Image.open(files['fitted_mask.png']) as fitted:
        if source.size != (512, 512) or fitted.size != source.size:
            raise ValueError('Require original512 masks')
        box = ImageChops.lighter(source.convert('L'), fitted.convert('L')).getbbox()
    if box is None:
        raise ValueError('Empty union foreground')
    crop = [max(0, box[0] - 16), max(0, box[1] - 16), min(512, box[2] + 16), min(512, box[3] + 16)]
    materials = {method: read(maps['material_outputs'][method]) for method in ('supermat', 'trellis')}
    if any(sha256(maps['material_outputs'][method]) != renders['inputs'][method]['sha256'] for method in materials):
        raise ValueError('Extracted material changed after rendering')
    if any(row[key] != materials[row['method']][key] for row in records for key in ('roughness', 'metallic')):
        raise ValueError('Render material differs from the extracted target')
    preparation = read(run / 'input/preparation.json')
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    report_git = {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain'))}
    output.mkdir(parents=True)
    for name, path in files.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if sha256(destination) != receipts[name]['sha256']:
            raise ValueError('Copied image hash mismatch')
    panel = output / 'panels'
    arm = [(output / 'original_mailbox.jpg', 'Original orange photograph'),
           (panel / 'supermat_albedo.png', 'SuperMat Albedo / original view'),
           (panel / 'supermat_roughness.png', 'SuperMat Roughness'),
           (panel / 'supermat_metallic.png', 'SuperMat Metallic'),
           (output / 'trellis_pbr.png', 'TRELLIS PBR / estimated view'),
           (panel / 'trellis_base_color.png', 'TRELLIS Base Color / estimated view'),
           (panel / 'trellis_roughness.png', 'TRELLIS Roughness'),
           (panel / 'trellis_metallic.png', 'TRELLIS Metallic')]
    sheet(arm, output / 'ARM.png', 4, crop)
    rendered = [(output / f"renders/{row['id']}.png", f"{row['shape']} / {row['color']} / {row['light']}\n"
                 f"{'SuperMat' if row['method'] == 'supermat' else 'TRELLIS.2'}  M={row['metallic']:.4f} R={row['roughness']:.4f}")
                for row in records]
    sheet(rendered, output / 'render_grid.png', 4)
    summary = {'schema': 'trellis_supermat_preview/v1', 'status': 'ready_for_user_review', 'training': False,
               'run': str(run), 'render_dir': str(render_dir),
               'git': {'input': preparation['git'], 'renders': renders['git_commit'], 'report': report_git},
               'materials': materials, 'coverage': maps['coverage'], 'camera_iou': maps['camera_iou'],
               'source': preparation, 'maps_report': maps, 'render_manifest': renders,
               'arm_crop': {'union_bbox': box, 'padding_pixels': 16, 'crop_bbox': crop,
                            'policy': 'identical crop/scale for all8 panels; originals retained'},
               'files': receipts, 'report_script_sha256': sha256(__file__)}
    for name in ('ARM.png', 'render_grid.png'):
        summary['files'][name] = {'sha256': sha256(output / name)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
    def card(name, title):
        name, title = html.escape(name, quote=True), html.escape(title)
        return f'<a target="_blank" href="{name}"><img loading="lazy" src="{name}"><span>{title}</span></a>'
    table = ''.join(f'<tr><td>{name}</td><td>{value["metallic"]:.6f}</td><td>{value["roughness"]:.6f}</td></tr>'
                    for name, value in materials.items())
    page = ('<!doctype html><meta charset="utf-8"><title>Mailbox A/R/M 对照</title><style>'
            'body{max-width:1500px;margin:24px auto;padding:0 20px;font:16px system-ui;background:#f5f6f8;color:#17212c}'
            'img{max-width:100%;display:block}a{color:inherit;text-decoration:none}span{display:block;padding:8px}'
            '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}'
            '.grid a{background:white;border:1px solid #ccd2db}table{border-collapse:collapse;margin:16px 0}'
            'td,th{padding:8px 24px;border:1px solid #bbc4cf}p{line-height:1.6}</style>'
            '<h1>Mailbox：SuperMat 与 TRELLIS.2</h1><p>同一橙色原图。点击任意图片打开完整分辨率。'
            '没有训练；共同 ROI 仅为估计视角下的图像重叠，不代表精确表面对应。</p>'
            '<p><a href="summary.json">完整数值、ROI、哈希与来源记录</a></p>'
            f'<table><tr><th>方法</th><th>Metallic</th><th>Roughness</th></tr>{table}</table>'
            '<h2>A / R / M</h2><p>两行使用相同联合前景边界裁切。第二行首图是估计视角下的 TRELLIS PBR；'
            '其余为属性展示图。PBR 不能用作 Albedo 真值。</p>' + card('ARM.png', '统一裁切对照') +
            '<h2>同形状、同灯光的标量材质对照</h2><p>每相邻两列为同一条件的 SuperMat / TRELLIS。'
            '完整画布等比缩放；只替换 M/R，Base Color、相机和灯光保持一致。</p>' + card('render_grid.png', '16张完整画布') +
            '<h2>原图与 ROI</h2><div class="grid">' + card('original_mailbox.jpg', '原图，未裁切') +
            card('reviewed_roi.png', '原 reviewed ROI') + card('common_roi.png', '共同 ROI') +
            ''.join(card('panels/' + name + '.png', name) for name in ('reviewed_roi_overlay', 'common_roi_overlay', 'trellis_roi_overlay')) +
            card('trellis_alpha.png', 'TRELLIS alpha') + '</div><h2>实际模型输入</h2>' +
            card('conditioning_comparison.png', '两种模型各自的官方预处理；内容源自同一原图') +
            '<h2>原始导出 A/R/M PNG</h2><p>TRELLIS PNG 为已有可视化导出；统计来自未量化浮点数据。</p><div class="grid">' +
            ''.join(card(f'{method}_{channel}.png', f'{method} {channel}') for method in ('supermat', 'trellis')
                    for channel in ('albedo', 'roughness', 'metallic')) + '</div><h2>完整渲染图</h2><div class="grid">' +
            ''.join(card(path.relative_to(output).as_posix(), title) for path, title in rendered) + '</div>')
    (output / 'index.html').write_text(page, encoding='utf8')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                package.write(path, path.relative_to(output).as_posix())
    return {'preview': str(output), 'index': str(output / 'index.html'), 'archive': str(archive)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-dir', type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.run, arguments.output, arguments.render_dir)))
