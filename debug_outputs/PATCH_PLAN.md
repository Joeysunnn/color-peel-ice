# Minimal compatibility patch

Move the existing logging directory value into
`ProjectConfiguration(logging_dir=..., total_limit=...)` and stop passing the
unsupported keyword directly to `Accelerator`.

Verification:

1. Static regression test locks the Accelerate 0.20.3 call shape.
2. Existing training-math and literal AdamW regression tests remain green.
3. Re-run the real two-step smoke from a new immutable run directory.

This is a compatibility fix, not a research contribution.

## Material evaluation bundle runtime fix

Do not patch the scientific code. Set `PYTHONPATH` to the immutable repository
root in the runtime campaign, record the compatibility flag in the launcher
environment, verify all three evaluation entry points import with `--help`, and
retry the non-resumable bundle stage under a fresh run ID. Preserve the failed
run as evidence.

## Material calibration reference-root fix

Keep the preparation code unchanged. Replace the overloaded reference-root
configuration with two explicit inputs: the accepted realized manifest and the
immutable renderer image root. Add a config regression assertion and retry the
non-resumable preparation stage under a new run ID.
