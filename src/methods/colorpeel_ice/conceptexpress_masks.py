"""Inference-only adaptation of ConceptExpress's step-zero localization.

Upstream: haoosz/ConceptExpress, db1dc61665866d208ba3a2e2152a83e7c6829e2f.
The pinned external checkout supplies FINCH and its distance/connectivity helpers;
the training entrypoint is never imported. FINCH has an academic-use notice.

Adapted attention/localization code: Copyright (c) 2023 Shaozhe Hao, MIT License.
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import ast
import hashlib
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image


UPSTREAM_COMMIT = 'db1dc61665866d208ba3a2e2152a83e7c6829e2f'


def load_finch(checkout):
    """Execute unchanged pinned functions without importing unrelated training deps."""
    checkout = Path(checkout).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(checkout), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != UPSTREAM_COMMIT or git('status', '--porcelain'):
        raise ValueError('ConceptExpress requires the pinned, clean upstream checkout')
    paths = [checkout / 'ptp_utils.py', checkout / 'clustering' / 'finch.py']
    sources = [path.read_text(encoding='utf8') for path in paths]
    namespace = {'np': np, '__name__': '_colorpeel_conceptexpress_finch'}
    # These are the only ptp_utils functions used by FINCH(distance='kld').
    names = {'kld_distance', 'get_num_mask', 'get_connect'}
    helpers = ast.parse(sources[0], filename=str(paths[0]))
    helpers.body = [node for node in helpers.body if isinstance(node, ast.FunctionDef)
                    and node.name in names]
    if {node.name for node in helpers.body} != names:
        raise ValueError('missing pinned FINCH helpers')
    exec(compile(helpers, str(paths[0]), 'exec'), namespace)
    finch = ast.parse(sources[1], filename=str(paths[1]))
    # Remove just its broad helper import; preserve the algorithm verbatim.
    finch.body = [node for node in finch.body if not
                 (isinstance(node, ast.ImportFrom) and node.module == 'ptp_utils')]
    exec(compile(finch, str(paths[1]), 'exec'), namespace)
    hashes = {str(path.relative_to(checkout)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    return namespace['FINCH'], hashes


def select_precluster_index(counts):
    for index in range(len(counts) - 1):
        if counts[index] >= 10 and counts[index + 1] < 10:
            return index
    raise ValueError(f'no official precluster crossing of 10: {counts}')


def mask_to_native(mask, size):
    mask = np.asarray(mask)
    if mask.shape != (64, 64) or not set(np.unique(mask).tolist()) <= {0, 1}:
        raise ValueError('expected a binary 64x64 latent mask')
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
        size, Image.Resampling.NEAREST)) != 0


class AttentionCapture:
    """Preserve upstream's last-half-head selection at batch size one."""

    def __init__(self, store, location):
        self.store, self.location = store, location

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        import torch
        batch, sequence, _ = hidden_states.shape
        if batch != 1:
            raise ValueError('localization supports one image at a time')
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence, batch)
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key = attn.head_to_batch_dim(attn.to_k(context))
        value = attn.head_to_batch_dim(attn.to_v(context))
        probs = attn.get_attention_scores(query, key, attention_mask)
        selected = probs[probs.shape[0] // 2:]
        cross = encoder_hidden_states is not None
        if not cross or (sequence == 16 ** 2 and self.location in ('up', 'down')):
            self.store.append((self.location, cross, selected.shape[0],
                               selected.mean(0).detach().cpu()))
        result = torch.bmm(probs, value)
        result = attn.batch_to_head_dim(result)
        return attn.to_out[1](attn.to_out[0](result))


def aggregate_attention(store, eot_index):
    import torch
    from torch.nn import functional as F
    aggregate, total_weight, eot_sum, cross_heads = None, 0., None, 0
    for location, cross, heads, attention in store:
        if cross:
            value = attention[:, eot_index].reshape(16, 16) * heads
            eot_sum = value if eot_sum is None else eot_sum + value
            cross_heads += heads
            continue
        res = int(attention.shape[-1] ** 0.5)
        aligned = F.interpolate(attention.reshape(1, -1, res, res),
                                size=(64, 64), mode='bilinear')
        aligned = aligned.reshape(1, res, res, -1)
        aligned = aligned.repeat_interleave(64 // res, dim=1).repeat_interleave(64 // res, dim=2)
        aligned = aligned.reshape(4096, 4096) * (res / 64)
        aggregate = aligned if aggregate is None else aggregate + aligned
        total_weight += res / 64
    if aggregate is None or eot_sum is None:
        raise ValueError('missing SD2.1 self or EOT cross-attention')
    features = aggregate / total_weight
    features = features / features.sum(-1, keepdim=True)
    eot = F.interpolate((eot_sum / cross_heads)[None, None], size=(64, 64),
                        mode='bilinear')[0, 0]
    if not torch.isfinite(features).all() or not (features > 0).all():
        raise ValueError('invalid attention distributions for upstream KL distance')
    if not torch.isfinite(eot).all() or eot.max() == eot.min():
        raise ValueError('constant or invalid EOT attention cannot filter foreground')
    eot = (eot - eot.min()) / (eot.max() - eot.min())
    return features.numpy(), eot.numpy()


def cluster_attention(features, eot, finch):
    labels, counts, _, thresholds = finch(features, distance='kld',
                                          ensure_early_exit=False, verbose=True)
    index = select_precluster_index(counts)
    selected = labels[:, index]
    candidates, means, scores = [], [], []
    for label in range(int(selected.max()) + 1):
        mask = (selected == label).reshape(64, 64)
        score = float(eot[mask].mean() / eot.mean())
        scores.append({'label': label, 'score': score, 'retained': score > 1.0})
        if score > 1.0:
            candidates.append(mask)
            means.append(features[selected == label].mean(0))
    if not candidates:
        raise ValueError('no foreground candidate passes the official EOT filter')
    candidates = np.stack(candidates)
    post, post_counts, _, _ = finch(np.stack(means), distance='kld',
        ensure_early_exit=True, verbose=True, mask_candidate=candidates.astype(np.float32),
        min_sim=thresholds[index])
    post = post[:, -1]
    masks = [candidates[post == label].any(0) for label in range(int(post.max()) + 1)]
    diagnostics = {'precluster_counts': [int(n) for n in counts],
        'precluster_index': index, 'distance_threshold': None if thresholds[index] is None
        else float(thresholds[index]), 'candidate_scores': scores,
        'postcluster_counts': [int(n) for n in post_counts],
        'precluster_labels': selected.reshape(64, 64), 'eot64': eot}
    return masks, diagnostics


def extract(image, model_dir, upstream_checkout, device='cuda:0', seed=20):
    """Return masks and diagnostics (eot64/precluster_labels are arrays); no writes."""
    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    finch, hashes = load_finch(upstream_checkout)
    tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder='tokenizer', local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(model_dir, subfolder='text_encoder',
                                                local_files_only=True).to(device).eval()
    vae = AutoencoderKL.from_pretrained(model_dir, subfolder='vae', local_files_only=True).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder='unet',
                                               local_files_only=True).to(device).eval()
    store = []
    processors = {}
    for name in unet.attn_processors:
        location = name.split('_')[0]
        if location not in ('down', 'up', 'mid'):
            raise ValueError(f'unsupported SD attention location: {name}')
        processors[name] = AttentionCapture(store, location)
    unet.set_attn_processor(processors)
    pixels = np.asarray(image.convert('RGB').resize((512, 512), Image.Resampling.BILINEAR)).copy()
    pixels = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device, torch.float32) / 127.5 - 1
    ids = tokenizer([''], padding='max_length', truncation=True,
                    max_length=tokenizer.model_max_length, return_tensors='pt').input_ids.to(device)
    eot_positions = (ids[0] == 49407).nonzero().flatten()
    if len(eot_positions) != 1:
        raise ValueError('expected SD2.1 tokenizer with a single EOT token 49407')
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        latent = vae.encode(pixels).latent_dist.sample(generator=generator) * 0.18215
        unet(latent, torch.zeros(1, dtype=torch.long, device=device), text_encoder(ids)[0])
    del unet, vae, text_encoder, pixels, latent
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    features, eot = aggregate_attention(store, int(eot_positions[0]))
    del store
    masks, diagnostics = cluster_attention(features, eot, finch)
    diagnostics.update(upstream_commit=UPSTREAM_COMMIT, upstream_files_sha256=hashes,
        model_dir=str(model_dir), seed=seed, device=device, dtype='float32',
        timestep=0, prompt='', input_size=list(image.size), model_input_size=[512, 512],
        latent_mask_size=[64, 64], flip=False, attention_heads='last_half_as_upstream',
        training_started=False,
        adaptation='inference only; explicit latent seed; no flip; eval mode; CPU aggregation')
    return masks, diagnostics
