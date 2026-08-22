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

