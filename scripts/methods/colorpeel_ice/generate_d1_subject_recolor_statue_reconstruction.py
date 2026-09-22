"""Generate fixed training-prompt reconstructions for three statue-subject checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image


MODEL_ID = "CompVis/stable-diffusion-v1-4"
WEIGHTS = "pytorch_custom_diffusion_weights.bin"
TOKEN_LOCAL_KV_WEIGHTS = "pytorch_token_local_kv_weights.bin"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_manifest(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    sampling_variants = protocol.get("sampling_variants")
    if sampling_variants is None:
        sampling_variants = [{"id": None, **protocol["sampling"]}]
    expected_count = protocol.get("expected_image_count")
    if expected_count is None:
        expected_count = protocol["sampling"]["expected_image_count"]
    for checkpoint in protocol["source_checkpoints"]:
        for sampling in sampling_variants:
            for item in protocol["prompts"]:
                for seed in sampling["seeds"]:
                    step = checkpoint["steps"]
                    checkpoint_id = checkpoint.get("id", f"step-{step}")
                    image_group = checkpoint.get("id", f"step_{step}")
                    sampling_id = sampling["id"]
                    sampling_group = "" if sampling_id is None else f"/{sampling_id}"
                    sampling_label = "" if sampling_id is None else f"-{sampling_id}"
                    rows.append({
                        "id": f"{checkpoint_id}{sampling_label}-{item['color']}-seed-{seed}", "checkpoint_steps": step,
                        "checkpoint_id": checkpoint_id, "sampling_id": sampling_id,
                        "model_dir": checkpoint["model_dir"], "color": item["color"], "prompt": item["prompt"],
                        "seed": seed, "num_inference_steps": sampling["num_inference_steps"],
                        "guidance_scale": sampling["guidance_scale"],
                        "image_path": f"images/{image_group}{sampling_group}/{item['color']}-seed-{seed}.png",
                    })
    if len(rows) != expected_count:
        raise ValueError("protocol expected_image_count does not match its grid")
    return rows


def validate_model_dir(path: Path, protocol: dict[str, Any]) -> dict[str, str]:
    resolved = path.resolve()
    checkpoint = next((item for item in protocol["source_checkpoints"] if Path(item["model_dir"]).resolve() == resolved), None)
    if checkpoint is None:
        raise ValueError("model directory is not bound by the protocol")
    token_artifacts = tuple(protocol.get("required_token_artifacts", ("<S*>.bin",)))
    weights_name = checkpoint.get("weight_name", WEIGHTS)
    required = (*token_artifacts, weights_name, "embedding_update_audit.json", "training_metrics.jsonl")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoint artifacts: " + ", ".join(missing))
    forbidden = [name for name in protocol["forbidden_token_artifacts"] if (path / name).exists()]
    if forbidden:
        raise ValueError("forbidden token artifacts: " + ", ".join(forbidden))
    hashes = {name: sha256(path / name) for name in required}
    if checkpoint.get("model_sha256") and hashes[weights_name] != checkpoint["model_sha256"]:
        raise ValueError("checkpoint weights do not match the protocol hash")
    for name, expected in checkpoint.get("token_artifact_sha256", {}).items():
        if name not in token_artifacts or hashes.get(name) != expected:
            raise ValueError(f"checkpoint token artifact does not match the protocol hash: {name}")
    if checkpoint.get("run_manifest_sha256"):
        run_dir = Path(checkpoint["run_dir"])
        if path.parent.resolve() != run_dir.resolve():
            raise ValueError("model directory does not belong to the protocol run directory")
        manifest = run_dir / "manifest.json"
        if not manifest.is_file() or sha256(manifest) != checkpoint["run_manifest_sha256"]:
            raise ValueError("run manifest does not match the protocol hash")
    if checkpoint.get("adaptation_mode") == "token_local_kv":
        adaptation = path / "adaptation_config.json"
        if not adaptation.is_file() or read_json(adaptation).get("adaptation_mode") != "token_local_kv":
            raise ValueError("token-local checkpoint is missing its adaptation metadata")
    return hashes


def load_pipeline(model_dir: Path, protocol: dict[str, Any], args: argparse.Namespace) -> Any:
    import torch
    from diffusers import DiffusionPipeline

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    checkpoint = next(item for item in protocol["source_checkpoints"] if Path(item["model_dir"]).resolve() == model_dir.resolve())
    if checkpoint.get("adaptation_mode") == "token_local_kv":
        train_root = str(Path(__file__).resolve().parents[3] / "src" / "train")
        if train_root not in sys.path:
            sys.path.insert(0, train_root)
        from custom_attention.unet_2d_condition_custom import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", local_files_only=True, torch_dtype=dtype
        )
        unet.load_attn_procs(
            str(model_dir),
            weight_name=checkpoint.get("weight_name", TOKEN_LOCAL_KV_WEIGHTS),
            adaptation_mode="token_local_kv",
        )
        pipe = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, unet=unet, low_cpu_mem_usage=False, torch_dtype=dtype, local_files_only=True
        ).to(args.device)
    else:
        pipe = DiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, low_cpu_mem_usage=False, torch_dtype=dtype, local_files_only=True).to(args.device)
        pipe.unet.load_attn_procs(str(model_dir), weight_name=WEIGHTS)
    for token_artifact in protocol.get("required_token_artifacts", ("<S*>.bin",)):
        pipe.load_textual_inversion(str(model_dir), weight_name=token_artifact)
    return pipe


def token_local_cross_attention_kwargs(pipe: Any, prompt: str, guidance_scale: float) -> dict[str, Any]:
    import torch

    modifier_id = pipe.tokenizer.convert_tokens_to_ids("<S*>")
    conditional_ids = pipe.tokenizer(
        prompt, padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length, return_tensors="pt"
    ).input_ids
    conditional_mask = conditional_ids == modifier_id
    if guidance_scale > 1.0:
        unconditional_ids = pipe.tokenizer(
            "", padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids
        conditional_mask = torch.cat([unconditional_ids == modifier_id, conditional_mask], dim=0)
    return {"modifier_token_mask": conditional_mask.to(pipe.unet.device)}


def replace_alignit_subject_slot(key, value, source_key, source_value, subject_index: int, conditional_only: bool = True):
    """Copy one source K/V token into base K/V without changing any other slot."""
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("AlignIT base K/V must have matching [batch, sequence, feature] shapes")
    if not 0 <= subject_index < key.shape[1]:
        raise ValueError("AlignIT subject index is outside the base prompt sequence")
    if source_key.shape != source_value.shape or source_key.ndim != 2 or source_key.shape[1] != key.shape[2]:
        raise ValueError("AlignIT source K/V must have matching [batch, feature] shapes")
    start = key.shape[0] // 2 if conditional_only and key.shape[0] > 1 else 0
    copied_key, copied_value = key.clone(), value.clone()
    copied_key[start:, subject_index] = source_key[:1].to(device=key.device, dtype=key.dtype).expand(key.shape[0] - start, -1)
    copied_value[start:, subject_index] = source_value[:1].to(device=value.device, dtype=value.dtype).expand(value.shape[0] - start, -1)
    return copied_key, copied_value


class AlignITKVAttnProcessor:
    """Base attention with one conditional cross-attention K/V token replaced at inference."""

    def __init__(self):
        self.source_key = None
        self.source_value = None
        self.subject_index = None
        self.last_ordinary_slots_unchanged = None

    def configure(self, source_key, source_value, subject_index: int) -> None:
        self.source_key = source_key.detach()
        self.source_value = source_value.detach()
        self.subject_index = int(subject_index)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if self.source_key is not None:
            if encoder_hidden_states is hidden_states:
                raise ValueError("AlignIT K/V replacement is valid only for cross-attention")
            base_key, base_value = key, value
            key, value = replace_alignit_subject_slot(key, value, self.source_key, self.source_value, self.subject_index)
            ordinary = [slot for slot in range(key.shape[1]) if slot != self.subject_index]
            self.last_ordinary_slots_unchanged = bool(
                (key[:, ordinary] == base_key[:, ordinary]).all() and (value[:, ordinary] == base_value[:, ordinary]).all()
            )
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def _single_token_id(tokenizer, text: str, field: str) -> int:
    token_ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(token_ids) != 1:
        raise ValueError(f"AlignIT {field} must tokenize to exactly one token: {text!r}")
    return int(token_ids[0])


def build_alignit_input_ids(tokenizer, prompt: str, modifier_token: str, class_token: str, dummy_policy: str):
    """Build target-base and source-dummy token sequences with an identical subject slot."""
    import torch

    target_ids = tokenizer(
        prompt, padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt"
    ).input_ids
    subject_id = tokenizer.convert_tokens_to_ids(modifier_token)
    positions = (target_ids[0] == subject_id).nonzero(as_tuple=False).flatten().tolist()
    if len(positions) != 1:
        raise ValueError("AlignIT requires exactly one explicit modifier-token position in every prompt")
    subject_index = int(positions[0])
    class_id = _single_token_id(tokenizer, class_token, "class token")
    filler_id = _single_token_id(tokenizer, "*", "dummy placeholder")
    base_ids = target_ids.clone()
    base_ids[0, subject_index] = class_id
    dummy_ids = base_ids.clone()
    protected_ids = {int(value) for value in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) if value is not None}
    for index, token_id in enumerate(dummy_ids[0].tolist()):
        if index != subject_index and token_id not in protected_ids:
            dummy_ids[0, index] = filler_id
    dummy_ids[0, subject_index] = subject_id
    if dummy_policy == "subject_phrase":
        phrase_index = subject_index + 1
        if phrase_index >= dummy_ids.shape[1] or base_ids[0, phrase_index].item() != class_id:
            raise ValueError("subject_phrase AlignIT requires the class token immediately after the modifier token")
        dummy_ids[0, phrase_index] = class_id
    elif dummy_policy != "strict":
        raise ValueError(f"unknown AlignIT dummy policy: {dummy_policy}")
    return base_ids, dummy_ids, subject_index


def encode_input_ids(pipe: Any, input_ids):
    import torch

    with torch.no_grad():
        return pipe.text_encoder(input_ids.to(pipe.unet.device))[0]


def collect_alignit_source_kv(custom_pipe: Any, dummy_ids, subject_index: int) -> dict[str, tuple[Any, Any]]:
    train_root = str(Path(__file__).resolve().parents[3] / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.attention_processor_custom import CustomDiffusionAttnProcessor, TokenLocalKVAttnProcessor

    source_hidden_states = encode_input_ids(custom_pipe, dummy_ids)
    source = {}
    for name, processor in custom_pipe.unet.attn_processors.items():
        if name.endswith("attn1.processor"):
            continue
        attention = custom_pipe.unet.get_submodule(name.removesuffix(".processor"))
        hidden_states = attention.norm_encoder_hidden_states(source_hidden_states) if attention.norm_cross else source_hidden_states
        if isinstance(processor, TokenLocalKVAttnProcessor):
            modifier_id = custom_pipe.tokenizer.convert_tokens_to_ids("<S*>")
            modifier_mask = dummy_ids.to(custom_pipe.unet.device) == modifier_id
            key, value = processor.project_kv(attention, hidden_states, modifier_mask)
        elif isinstance(processor, CustomDiffusionAttnProcessor) and processor.train_kv:
            key = processor.to_k_custom_diffusion(hidden_states)
            value = processor.to_v_custom_diffusion(hidden_states)
        else:
            raise TypeError(f"AlignIT source requires token-local or full-K/V cross-attention processors: {name}")
        source[name] = (key[:, subject_index].detach(), value[:, subject_index].detach())
    return source


def configure_alignit_base_unet(base_unet, source_kv: dict[str, tuple[Any, Any]], subject_index: int) -> None:
    for name, (source_key, source_value) in source_kv.items():
        processor = base_unet.attn_processors.get(name)
        if not isinstance(processor, AlignITKVAttnProcessor):
            raise TypeError(f"AlignIT base processor is missing for cross-attention layer: {name}")
        processor.configure(source_key, source_value, subject_index)


def load_alignit_base_pipeline(args: argparse.Namespace) -> Any:
    import torch
    from diffusers import DiffusionPipeline

    train_root = str(Path(__file__).resolve().parents[3] / "src" / "train")
    if train_root not in sys.path:
        sys.path.insert(0, train_root)
    from custom_attention.attention_processor_custom import AttnProcessor
    from custom_attention.unet_2d_condition_custom import UNet2DConditionModel

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", local_files_only=True, torch_dtype=dtype
    )
    processors = {
        name: AttnProcessor() if name.endswith("attn1.processor") else AlignITKVAttnProcessor()
        for name in unet.attn_processors
    }
    unet.set_attn_processor(processors)
    return DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, unet=unet, low_cpu_mem_usage=False, torch_dtype=dtype, local_files_only=True
    ).to(args.device)


def alignit_prompt_embeds(base_pipe: Any, base_ids, guidance_scale: float):
    prompt_embeds = encode_input_ids(base_pipe, base_ids)
    if guidance_scale <= 1.0:
        return prompt_embeds, None
    unconditioned_ids = base_pipe.tokenizer(
        "", padding="max_length", truncation=True, max_length=base_pipe.tokenizer.model_max_length, return_tensors="pt"
    ).input_ids
    return prompt_embeds, encode_input_ids(base_pipe, unconditioned_ids)


def generate_alignit_image(custom_pipe: Any, base_pipe: Any, checkpoint: dict[str, Any], row: dict[str, Any], args: argparse.Namespace):
    alignit = checkpoint.get("alignit")
    if not isinstance(alignit, dict):
        raise ValueError("AlignIT checkpoint metadata is required")
    base_ids, dummy_ids, subject_index = build_alignit_input_ids(
        custom_pipe.tokenizer,
        row["prompt"],
        modifier_token=alignit.get("modifier_token", "<S*>"),
        class_token=alignit.get("class_token", "mailbox"),
        dummy_policy=alignit["dummy_policy"],
    )
    source_kv = collect_alignit_source_kv(custom_pipe, dummy_ids, subject_index)
    configure_alignit_base_unet(base_pipe.unet, source_kv, subject_index)
    prompt_embeds, negative_prompt_embeds = alignit_prompt_embeds(base_pipe, base_ids, row["guidance_scale"])
    result = base_pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=row["num_inference_steps"],
        guidance_scale=row["guidance_scale"],
        generator=torch.Generator(device=args.device).manual_seed(row["seed"]),
    )
    ordinary_slots_unchanged = [
        processor.last_ordinary_slots_unchanged
        for name, processor in base_pipe.unet.attn_processors.items()
        if not name.endswith("attn1.processor")
    ]
    if not ordinary_slots_unchanged or not all(ordinary_slots_unchanged):
        raise RuntimeError("AlignIT invariant failed: a non-subject base K/V slot changed")
    return result, {
        "class_token": alignit.get("class_token", "mailbox"),
        "dummy_policy": alignit["dummy_policy"],
        "subject_index": subject_index,
        "base_token_ids": base_ids[0].tolist(),
        "dummy_token_ids": dummy_ids[0].tolist(),
        "cross_attention_layer_count": len(source_kv),
        "ordinary_base_kv_exact": True,
    }


def all_black(image: Image.Image) -> bool:
    return image.convert("RGB").getextrema() == ((0, 0), (0, 0), (0, 0))


def generate(rows: list[dict[str, Any]], protocol: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    statuses, artifacts = [], {}
    for model_dir_text in dict.fromkeys(row["model_dir"] for row in rows):
        model_dir = Path(model_dir_text)
        artifacts[str(model_dir)] = validate_model_dir(model_dir, protocol)
        checkpoint = next(item for item in protocol["source_checkpoints"] if item["model_dir"] == model_dir_text)
        inference_mode = checkpoint.get("inference_mode", "token_local")
        pipe = load_pipeline(model_dir, protocol, args)
        base_pipe = load_alignit_base_pipeline(args) if inference_mode == "alignit" else None
        for row in (item for item in rows if item["model_dir"] == model_dir_text):
            path = args.output_dir / row["image_path"]
            status = {"id": row["id"], "image_path": str(path), "status": None, "failure_reason": None, "image_sha256": None, "nsfw_content_detected": None, "alignit": None}
            try:
                if inference_mode == "alignit":
                    result, status["alignit"] = generate_alignit_image(pipe, base_pipe, checkpoint, row, args)
                elif inference_mode == "token_local":
                    kwargs = {}
                    if checkpoint.get("adaptation_mode") == "token_local_kv":
                        kwargs["cross_attention_kwargs"] = token_local_cross_attention_kwargs(
                            pipe, row["prompt"], row["guidance_scale"]
                        )
                    result = pipe(row["prompt"], num_inference_steps=row["num_inference_steps"], guidance_scale=row["guidance_scale"], generator=torch.Generator(device=args.device).manual_seed(row["seed"]), **kwargs)
                else:
                    raise ValueError(f"unknown inference_mode: {inference_mode}")
                image = result.images[0]
                detected = getattr(result, "nsfw_content_detected", None)
                status["nsfw_content_detected"] = bool(detected[0]) if isinstance(detected, (list, tuple)) and detected else False
                if not isinstance(image, Image.Image) or image.mode != "RGB" or image.size != (512, 512):
                    raise ValueError("pipeline must return a 512x512 RGB PIL image")
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                status["image_sha256"] = sha256(path)
                if status["nsfw_content_detected"]:
                    status.update(status="failure", failure_reason="safety_checker_filtered")
                elif all_black(image):
                    status.update(status="failure", failure_reason="all_black_output")
                else:
                    status["status"] = "ok"
            except Exception as error:
                status.update(status="failure", failure_reason=f"generation_error:{type(error).__name__}:{error}")
            statuses.append(status)
        if base_pipe is not None:
            del base_pipe
        del pipe
        torch.cuda.empty_cache()
    return statuses, artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--pretrained-model-name-or-path", default=MODEL_ID)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = read_json(args.protocol)
    rows = build_manifest(protocol)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, args.output_dir / "generation_manifest.jsonl")
    provenance = {"protocol_path": str(args.protocol.resolve()), "protocol_sha256": sha256(args.protocol), "model_artifact_sha256": None}
    if args.dry_run:
        write_json(args.output_dir / "provenance.json", provenance)
        return 0
    statuses, artifact_hashes = generate(rows, protocol, args)
    provenance["model_artifact_sha256"] = artifact_hashes
    write_json(args.output_dir / "provenance.json", provenance)
    write_jsonl(statuses, args.output_dir / "generation_status.jsonl")
    return 0 if all(row["status"] == "ok" for row in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
