"""SDXL dual CLIP conditioning with only two new token vectors trainable."""

import torch
from torch import nn


class FrozenTokenEmbedding(nn.Module):
    """Keep the complete original vocabulary frozen and optimize one extra row."""

    def __init__(self, base, token_id, initializer_id):
        super().__init__()
        if token_id != base.num_embeddings:
            raise ValueError("the learned token must be the only newly appended token")
        self.base = base.requires_grad_(False)
        self.vector = nn.Parameter(base.weight[initializer_id].detach().float().clone())
        self.token_id = token_id
        self.num_embeddings = base.num_embeddings + 1
        self.embedding_dim = base.embedding_dim

    @property
    def weight(self):
        # Compatibility with CLIP tooling; ordinary forward avoids this large concat.
        return torch.cat((self.base.weight, self.vector.to(self.base.weight.dtype).unsqueeze(0)))

    def forward(self, input_ids):
        mask = input_ids == self.token_id
        ordinary = self.base(input_ids.masked_fill(mask, 0))
        return torch.where(mask.unsqueeze(-1), self.vector.to(ordinary.dtype), ordinary)


def fix_eos_pooling(tokenizers, text_encoders):
    """Avoid CLIP's legacy argmax pooling selecting the appended token as EOS."""
    for tokenizer, encoder in zip(tokenizers, text_encoders):
        encoder.config.eos_token_id = tokenizer.eos_token_id
        encoder.text_model.eos_token_id = tokenizer.eos_token_id


def install_learned_tokens(tokenizers, text_encoders, token, initializer):
    if len(tokenizers) != 2 or len(text_encoders) != 2:
        raise ValueError("SDXL requires both tokenizers and text encoders")
    initializers = [tokenizer.encode(initializer, add_special_tokens=False) for tokenizer in tokenizers]
    if any(len(ids) != 1 for ids in initializers):
        raise ValueError("initializer must be exactly one token in both SDXL tokenizers")
    if any(token in tokenizer.get_vocab() for tokenizer in tokenizers):
        raise ValueError("learned token already exists; start each branch from the base model")
    embeddings = []
    for tokenizer, encoder, initializer_ids in zip(tokenizers, text_encoders, initializers):
        encoder.requires_grad_(False)
        if tokenizer.add_tokens(token) != 1:
            raise ValueError("could not append one learned token")
        embedding = FrozenTokenEmbedding(
            encoder.get_input_embeddings(), tokenizer.convert_tokens_to_ids(token), initializer_ids[0],
        )
        encoder.set_input_embeddings(embedding)
        encoder.config.vocab_size = len(tokenizer)
        embeddings.append(embedding)
    fix_eos_pooling(tokenizers, text_encoders)
    return embeddings


def encode_conditioning(tokenizers, text_encoders, prompts, token=None):
    """Return penultimate concat, projected CLIP-G EOS, and the token-position gate.

    No no_grad decorator: gradients must reach the two learned vectors through
    both contextual hidden states and the global pooled conditioning pathway.
    No attention_mask is passed, matching native StableDiffusionXLPipeline.
    """
    if len(tokenizers) != 2 or len(text_encoders) != 2:
        raise ValueError("SDXL requires both tokenizers and text encoders")
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    fix_eos_pooling(tokenizers, text_encoders)
    hidden_states, masks = [], []
    for tokenizer, encoder in zip(tokenizers, text_encoders):
        tokenized = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length,
                              truncation=True, return_tensors="pt")
        input_ids = tokenized.input_ids.to(next(encoder.parameters()).device)
        if not (input_ids == tokenizer.eos_token_id).any(dim=1).all():
            raise ValueError("every prompt must contain an EOS token")
        if token is None:
            mask = torch.zeros_like(input_ids, dtype=torch.bool)
        else:
            if token not in tokenizer.get_vocab():
                raise ValueError("target token is absent from tokenizer")
            mask = input_ids == tokenizer.convert_tokens_to_ids(token)
            for prompt, positions in zip(prompts, mask):
                if prompt.count(token) != int(positions.sum()):
                    raise ValueError("target token was truncated or not tokenized as an atomic token")
        output = encoder(input_ids, output_hidden_states=True, return_dict=True)
        hidden_states.append(output.hidden_states[-2])
        masks.append(mask)
    if not torch.equal(masks[0].cpu(), masks[1].cpu()):
        raise ValueError("the two SDXL tokenizers disagree on target-token positions")
    return torch.cat(hidden_states, dim=-1), output.text_embeds, masks[0]


def make_time_ids(original_size, target_size, crop=(0, 0), batch_size=1, device=None, dtype=None):
    """SDXL order: original height/width, crop top/left, target height/width."""
    if any(len(pair) != 2 for pair in (original_size, crop, target_size)):
        raise ValueError("original_size, crop and target_size must each contain two values")
    values = tuple(original_size) + tuple(crop) + tuple(target_size)
    if any(value <= 0 for value in tuple(original_size) + tuple(target_size)) or any(value < 0 for value in crop):
        raise ValueError("invalid SDXL size/crop conditioning")
    return torch.tensor([values], device=device, dtype=dtype or torch.float32).repeat(batch_size, 1)
