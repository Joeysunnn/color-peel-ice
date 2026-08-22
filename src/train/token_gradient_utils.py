"""Small helpers for restricting text-embedding updates to modifier tokens."""


def modifier_rows_to_zero(vocab_size, modifier_token_ids):
    """Return a boolean mask that is true for every non-modifier vocab row."""
    modifier_ids = {int(token_id) for token_id in modifier_token_ids}
    if not modifier_ids:
        raise ValueError("at least one modifier token id is required")
    if min(modifier_ids) < 0 or max(modifier_ids) >= vocab_size:
        raise ValueError("modifier token id is outside the tokenizer vocabulary")
    return [index not in modifier_ids for index in range(vocab_size)]
