"""Validation helpers for textual-inversion initializer tokens."""


def single_token_initializer_id(tokenizer, initializer_token: str) -> int:
    """Return the token id, rejecting initializers split into multiple BPE pieces."""

    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Initializer token {initializer_token!r} must encode to exactly one token; "
            f"got {len(token_ids)} token ids: {token_ids}"
        )
    return int(token_ids[0])
