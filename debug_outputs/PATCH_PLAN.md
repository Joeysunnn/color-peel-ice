# Minimal compatibility patch

Move the existing logging directory value into
`ProjectConfiguration(logging_dir=..., total_limit=...)` and stop passing the
unsupported keyword directly to `Accelerator`.

Verification:

1. Static regression test locks the Accelerate 0.20.3 call shape.
2. Existing training-math and literal AdamW regression tests remain green.
3. Re-run the real two-step smoke from a new immutable run directory.

This is a compatibility fix, not a research contribution.

