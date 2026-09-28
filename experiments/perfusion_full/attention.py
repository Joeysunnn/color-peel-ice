"""Paper Eq. (3)/(4) Perfusion projections for one or two concepts."""

from __future__ import annotations

import torch
from torch import nn


class FullPerfusionAttnProcessor(nn.Module):
    """Edit all text positions; keep Key targets fixed and learn Value targets."""

    def __init__(self, hidden_size=None, cross_attention_dim=None,
                 key_outputs=None, value_outputs=None):
        super().__init__()
        if cross_attention_dim is None:
            if key_outputs is not None or value_outputs is not None:
                raise ValueError("self-attention cannot have Perfusion outputs")
            return
        if key_outputs is None or value_outputs is None:
            raise ValueError("cross-attention requires Key and Value target outputs")
        key_outputs = torch.as_tensor(key_outputs).detach().clone()
        value_outputs = torch.as_tensor(value_outputs).detach().clone()
        if (key_outputs.ndim != 2 or key_outputs.shape != value_outputs.shape
                or key_outputs.shape[0] not in (1, 2)
                or key_outputs.shape[1] != hidden_size):
            raise ValueError("target outputs must have shape [1 or 2, hidden_size]")
        self.register_buffer("key_outputs", key_outputs)
        self.value_outputs = nn.Parameter(value_outputs)
        self.cross_attention_dim = cross_attention_dim

    def project_kv(self, attn, encoder_hidden_states, inv_cov, anchors, beta, tau):
        if not hasattr(self, "key_outputs"):
            raise ValueError("self-attention has no Perfusion projections")
        if inv_cov is None or anchors is None or beta is None or tau is None:
            raise ValueError("Perfusion covariance, anchors, beta, and tau are required")
        if encoder_hidden_states.ndim != 3:
            raise ValueError("text encodings must have shape [batch, tokens, channels]")
        channels = encoder_hidden_states.shape[-1]
        concepts = self.key_outputs.shape[0]
        if (channels != self.cross_attention_dim or inv_cov.shape != (channels, channels)
                or anchors.shape != (concepts, channels)):
            raise ValueError("Perfusion covariance or anchor shape differs from text encodings")
        if not torch.isfinite(inv_cov).all() or not torch.isfinite(anchors).all():
            raise ValueError("Perfusion covariance and anchors must be finite")

        # For J concepts this is the C^-1 metric projection onto their joint span.
        # It is algebraically equivalent to Appendix B's Cholesky/QR construction,
        # while avoiding a 768x768 Cholesky factorization on every UNet call.
        with torch.cuda.amp.autocast(enabled=False):
            states = encoder_hidden_states.float()
            anchors = anchors.detach().to(device=states.device, dtype=torch.float32)
            inv_cov = inv_cov.detach().to(device=states.device, dtype=torch.float32)
            weighted_anchors = inv_cov @ anchors.T
            gram = anchors @ weighted_anchors
            similarities = states @ weighted_anchors
            energies = gram.diagonal()
            if torch.any(energies <= 0):
                raise ValueError("Perfusion anchors need positive covariance energy")
            if concepts == 2:
                overlap = gram[0, 1] / torch.sqrt(energies[0] * energies[1])
                if torch.abs(overlap) >= 1 - 1e-5:
                    raise ValueError("Perfusion anchors must span independent directions")
            coefficients = torch.linalg.solve(gram, similarities.reshape(-1, concepts).T).T
            coefficients = coefficients.reshape(*similarities.shape)
            orthogonal = states - coefficients @ anchors

            beta = torch.as_tensor(beta, device=states.device, dtype=torch.float32)
            tau = torch.as_tensor(tau, device=states.device, dtype=torch.float32)
            if (beta.ndim > 1 or tau.ndim > 1
                    or beta.numel() not in (1, concepts) or tau.numel() not in (1, concepts)
                    or not torch.isfinite(beta).all() or not torch.isfinite(tau).all()
                    or torch.any(tau <= 0)):
                raise ValueError("beta and positive tau must be scalar or per-concept")
            gates = torch.sigmoid((similarities / energies - beta) / tau)
        key = attn.to_k(orthogonal.to(encoder_hidden_states.dtype))
        value = attn.to_v(orthogonal.to(encoder_hidden_states.dtype))
        key = key + gates.to(key.dtype) @ self.key_outputs.to(key.dtype)
        value = value + gates.to(value.dtype) @ self.value_outputs.to(value.dtype)
        return key, value

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 perfusion_inv_cov=None, perfusion_anchors=None, perfusion_beta=None,
                 perfusion_tau=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            key, value = self.project_kv(
                attn, encoder_hidden_states, perfusion_inv_cov, perfusion_anchors,
                perfusion_beta, perfusion_tau,
            )
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        attn.attn_probs = attention_probs
        output = attn.batch_to_head_dim(torch.bmm(attention_probs, value))
        return attn.to_out[1](attn.to_out[0](output))
