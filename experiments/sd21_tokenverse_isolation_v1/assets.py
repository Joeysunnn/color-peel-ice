"""Fixed 25-image SD2.1 bank for the TokenVerse-inspired isolation experiment.

Captions, SD2.1, PNDM100, CFG3.5, 512px, fp16 and seeds1000..1024 are our
adaptation choices, not claimed paper settings. Captions are actual generation
prompts; image-caption agreement still requires the saved visual review.
"""

import hashlib
import html
import io
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


CAPTIONS = (
    'A cat resting on a cushion',
    'A dog running across a grassy field',
    'A horse standing beside a metal gate',
    'A penguin standing on dark rocks',
    'A turtle swimming underwater',
    'An elephant beside a watering hole',
    'A butterfly resting on a leaf',
    'A red bicycle leaning against a brick wall',
    'A blue car parked beside a warehouse',
    'A passenger train crossing a stone bridge',
    'A sailboat on a calm lake',
    'A helicopter above distant mountains',
    'A ceramic teapot on a kitchen counter',
    'A glass bottle on a white shelf',
    'A pair of shoes beside a suitcase',
    'A green backpack on a chair',
    'A silver wristwatch on a dark cushion',
    'A yellow umbrella beside a doorway',
    'A tall lighthouse above rocky cliffs',
    'A stone castle on a grassy hill',
    'A domed observatory beneath a starry sky',
    'A waterfall descending into a deep canyon',
    'A winding river between green mountains',
    'A volcano beneath a clear blue sky',
    'A desert dune at sunset',
)
SAMPLING = {'scheduler': 'PNDM', 'num_inference_steps': 100, 'guidance_scale': 3.5,
            'width': 512, 'height': 512, 'dtype': 'float16', 'safety_checker': True,
            'generator': 'fresh_device_generator_per_image', 'negative_prompt': None}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


def preflight(tokenizer, concept_prompts=None):
    """Count complete BOS/EOS-inclusive sequences; never truncate or add tokens."""
    if tokenizer.model_max_length != 77:
        raise ValueError('bank requires the SD2.1 CLIP77 tokenizer')
    left = [] if concept_prompts is None else list(concept_prompts)
    if isinstance(concept_prompts, str) or any(not isinstance(p, str) or not p.strip() for p in left):
        raise ValueError('concept_prompts must be a sequence of nonempty captions')
    from src.train.concept_isolation import combine_prompt
    standalone = [len(tokenizer.encode(p, add_special_tokens=True)) for p in CAPTIONS]
    combined = [[len(tokenizer.encode(combine_prompt(p, q), add_special_tokens=True)) for q in CAPTIONS] for p in left]
    if any(count > 77 for count in standalone) or any(count > 77 for row in combined for count in row):
        raise ValueError('bank or combined caption exceeds CLIP77 including BOS/EOS; no truncation allowed')
    return {'tokenizer_max_length': 77, 'includes_bos_eos': True, 'concept_prompts': left,
            'standalone_lengths': standalone, 'combined_lengths': combined,
            'combined_pairs_checked': len(left) * 25, 'combined_check_requested': concept_prompts is not None,
            'modifier_tokens_added': False,
            'note': 'Base tokenizer may split placeholders; training must recheck with its actual modifier token table.'}


def _render(output, records, status):
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 14)
    except OSError:
        font = ImageFont.load_default()
    sheet = Image.new('RGB', (5 * 224, 5 * 280 + 36), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), f'SD2.1 fixed25 bank | {status} | image-caption agreement requires human review', font=font, fill='black')
    by_id = {row['id']: row for row in records}
    cards = []
    for index, caption in enumerate(CAPTIONS):
        x, y = (index % 5) * 224 + 8, (index // 5) * 280 + 36
        row = by_id.get(index)
        label = 'not generated' if row is None else row['status']
        draw.text((x, y), f'{index:02d} seed{1000 + index} {label}', font=font, fill='black')
        if row is not None:
            with Image.open(row['path']) as image:
                sheet.paste(image.resize((200, 200), Image.Resampling.LANCZOS), (x, y + 22))
            url = Path(row['path']).relative_to(output).as_posix()
            picture = f'<a href="{url}"><img src="{url}" width="200"></a>'
        else:
            draw.rectangle((x, y + 22, x + 199, y + 221), fill='#dddddd')
            picture = '<p>not generated</p>'
        draw.multiline_text((x, y + 225), textwrap.fill(caption, 29), font=font, fill='black', spacing=2)
        cards.append(f'<article><p>{index:02d} / seed{1000 + index} / {label}</p>{picture}'
                     f'<p>{html.escape(caption)}</p></article>')
    sheet.save(output / 'contact_sheet.png')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fixed25 bank</title>'
        '<style>body{font:16px sans-serif}main{display:grid;grid-template-columns:repeat(5,224px)}article{padding:8px}</style>'
        f'<h1>SD2.1 fixed25 bank: {html.escape(status)}</h1><p>Original generation captions; inspect each image-caption pair. '
        'No automatic resampling or automatic semantic acceptance.</p><p><a href="manifest.json">Manifest</a> | '
        '<a href="contact_sheet.png">Contact sheet</a></p><main>' + ''.join(cards) + '</main>', encoding='utf8')


def verify_bank(manifest):
    """Verify ready bank identity and bytes; do not assert visual caption accuracy."""
    path = Path(manifest).resolve()
    value = json.loads(path.read_text(encoding='utf8'))
    records = value.get('records', [])
    if (value.get('schema') != 1 or value.get('status') != 'ready' or value.get('sampling') != SAMPLING
            or len(records) != 25 or [row.get('id') for row in records] != list(range(25))):
        raise ValueError('require a ready schema1 bank with the exact25 ordered records and sampling')
    if not isinstance(value.get('model_receipt'), dict) or not value['model_receipt'] or not value.get('git'):
        raise ValueError('bank model receipt or generation git is missing')
    locks = {str(path): _sha256(path)}
    image_paths = set()
    for index, row in enumerate(records):
        image_path = Path(row.get('path', ''))
        if (not image_path.is_absolute() or image_path.is_symlink()
                or image_path.resolve() != path.parent / 'images' / f'{index:02d}.png'
                or row.get('caption') != CAPTIONS[index] or row.get('seed') != 1000 + index
                or row.get('status') != 'ok' or row.get('size') != [512, 512] or row.get('mode') != 'RGB'):
            raise ValueError(f'bank image identity/caption/status differs at id{index}')
        payload = image_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row.get('sha256'):
            raise ValueError(f'bank image SHA differs at id{index}')
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if (image.format != 'PNG' or image.size != (512, 512) or image.mode != 'RGB'
                    or all(high == 0 for low, high in image.getextrema())):
                raise ValueError(f'bank image must be nonblack PNG512 RGB at id{index}')
        locks[str(image_path)] = row['sha256']
        image_paths.add(image_path.resolve())
    if {item.resolve() for item in (path.parent / 'images').iterdir()} != image_paths:
        raise ValueError('bank image inventory differs from its exact25 records')
    return locks


def generate(output, concept_prompts=None):
    """Generate once on GPU0; caller owns GPU availability/model/source checks.

    A filtered or black image is saved for inspection and stops the bank. It is
    never accepted by verify_bank and is never replaced by another seed.
    """
    from experiments.sd21_wood_branches_v1 import inference as shared
    import torch

    output = Path(output).resolve()
    protected = (shared.ROOT.resolve(), shared.MODEL_DIR.resolve())
    if output.exists() or any(output.is_relative_to(root) or root.is_relative_to(output) for root in protected):
        raise ValueError('bank output must be fresh and separate from source/model directories')
    if any(SAMPLING[key] != value for key, value in shared.SAMPLING.items()):
        raise ValueError('shared sampling differs from the registered bank settings')
    receipt_path = shared.MODEL_DIR / 'model_source.json'
    receipt, receipt_hash = json.loads(receipt_path.read_text(encoding='utf8')), _sha256(receipt_path)
    revision = shared.git_info()
    pipe = shared.load_pipeline([], 'cuda:0', stock=True)
    state, rows = None, []
    try:
        length_check = preflight(pipe.tokenizer, concept_prompts)
        output.mkdir(parents=True)
        (output / 'images').mkdir()
        state = {'schema': 1, 'status': 'generating', 'records': rows, 'sampling': SAMPLING,
                 'model_receipt': receipt, 'model_receipt_path': str(receipt_path.resolve()),
                 'model_receipt_sha256': receipt_hash, 'git': revision, 'caption_length_check': length_check,
                 'caption_provenance': 'Exact generation prompts; not paper figure captions or post-hoc labels.',
                 'settings_provenance': 'SD2.1 sampling and captions are this experiment adaptation, not paper hyperparameters.',
                 'visual_caption_agreement_reviewed': False, 'automatic_resampling': False}
        _write(output / 'manifest.json', state)
        pipe.set_progress_bar_config(disable=True)
        with torch.no_grad(), (output / 'generation_status.jsonl').open('x', encoding='utf8') as ledger:
            for index, caption in enumerate(CAPTIONS):
                seed = 1000 + index
                result = pipe(caption, num_inference_steps=100, guidance_scale=3.5, width=512, height=512,
                              generator=torch.Generator(device='cuda:0').manual_seed(seed))
                if (result.nsfw_content_detected is None or len(result.nsfw_content_detected) != 1
                        or len(result.images) != 1 or result.images[0].size != (512, 512) or result.images[0].mode != 'RGB'):
                    raise ValueError('require one512 RGB image with an explicit safety result')
                image = result.images[0]
                status = 'safety_filtered' if result.nsfw_content_detected[0] else 'ok'
                if status == 'ok' and all(high == 0 for low, high in image.getextrema()):
                    status = 'black_image'
                path = output / 'images' / f'{index:02d}.png'
                with path.open('xb') as handle:
                    image.save(handle, format='PNG')
                row = {'id': index, 'path': str(path), 'sha256': _sha256(path), 'caption': caption,
                       'size': [512, 512], 'mode': 'RGB', 'seed': seed, 'status': status}
                rows.append(row)
                ledger.write(json.dumps(row) + '\n')
                ledger.flush()
                _write(output / 'manifest.json', state)
                if status != 'ok':
                    state['status'] = status
                    raise ValueError(f'bank stopped at id{index}: {status}; no replacement image is allowed')
        if shared.git_info() != revision or _sha256(receipt_path) != receipt_hash:
            raise ValueError('generation checkout or model receipt changed')
        state['status'] = 'ready'
        _write(output / 'manifest.json', state)
        locks = verify_bank(output / 'manifest.json')
        locks[str(receipt_path.resolve())] = receipt_hash
        locks[str(output / 'generation_status.jsonl')] = _sha256(output / 'generation_status.jsonl')
        return {'manifest_path': str(output / 'manifest.json'), 'source_locks': locks, 'count': 25}
    except BaseException as error:
        if state is not None:
            if state['status'] not in ('safety_filtered', 'black_image'):
                state['status'] = 'failed'
            state['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if state is not None:
            _write(output / 'manifest.json', state)
            _render(output, rows, state['status'])
        del pipe
        torch.cuda.empty_cache()
