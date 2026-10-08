def validate_checkpoint_plan(checkpoint_steps, checkpointing_steps, max_train_steps):
    if checkpointing_steps <= 0:
        raise ValueError("checkpointing_steps must be positive")
    if checkpoint_steps is None:
        return None
    if (not checkpoint_steps
            or any(not isinstance(step, int) or isinstance(step, bool) or step <= 0
                   for step in checkpoint_steps)
            or checkpoint_steps != sorted(set(checkpoint_steps))):
        raise ValueError("checkpoint_steps must be unique positive integers in increasing order")
    if max_train_steps is not None and checkpoint_steps[-1] > max_train_steps:
        raise ValueError("checkpoint_steps cannot exceed max_train_steps")
    return frozenset(checkpoint_steps)


def should_save_checkpoint(global_step, checkpoint_steps, checkpointing_steps):
    return (global_step in checkpoint_steps if checkpoint_steps is not None
            else global_step % checkpointing_steps == 0)
