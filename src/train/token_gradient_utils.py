"""Small helpers for restricting text-embedding updates to modifier tokens."""


def modifier_rows_to_zero(vocab_size, modifier_token_ids):
    """Return a boolean mask that is true for every non-modifier vocab row."""
    modifier_ids = {int(token_id) for token_id in modifier_token_ids}
    if not modifier_ids:
        raise ValueError("at least one modifier token id is required")
    if min(modifier_ids) < 0 or max(modifier_ids) >= vocab_size:
        raise ValueError("modifier token id is outside the tokenizer vocabulary")
    return [index not in modifier_ids for index in range(vocab_size)]


def inactive_modifier_ids_for_unpaired_step(input_ids, modifier_token_ids):
    """Require one exposed modifier and return all modifier rows absent this step."""
    present = [int(token_id) for token_id in modifier_token_ids
               if bool((input_ids == token_id).any().item())]
    if len(present) != 1:
        raise ValueError("unpaired modifier training requires exactly one learned token per step")
    return [int(token_id) for token_id in modifier_token_ids if token_id != present[0]]
