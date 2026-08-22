# Accelerate 0.20.3 startup diagnosis

- Failed run: real two-step smoke at commit `6adca6d`.
- Failure point: `Accelerator(...)` construction, before model loading and before
  any optimization step.
- Error: `Accelerator.__init__() got an unexpected keyword argument
  'logging_dir'`.
- Direct evidence: in pinned Accelerate 0.20.3, `Accelerator.__init__` accepts
  `project_config` but not `logging_dir`; `ProjectConfiguration` accepts
  `logging_dir` and `total_limit`.
- Cause category: non-scientific dependency API compatibility.
- Scientific impact: none expected. Loss, optimizer, trainable parameters,
  data order, gradients, and update order are unchanged.

## Resolution evidence

Commit `c8c874d00318ae7c1df2265c8627787d316a1ce3` was deployed by
`git pull --ff-only`. A fresh two-step smoke, a fresh nine-step coverage smoke,
and the full 1500-step run all completed successfully. The two smokes and full
run produced finite loss records, saved the Custom Diffusion weights and six
token files, and reloaded all six textual-inversion embeddings. This closes the
startup failure without changing training mathematics.
