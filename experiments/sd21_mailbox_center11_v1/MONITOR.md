# center11 server watchdog

The user authorized continuous monitoring and automatic repair after the original
startup. Run `python -B -u -m experiments.sd21_mailbox_center11_v1.monitor` from a
separate committed checkout, never by updating the running87d6e18 checkout.
The watchdog is scoped to the exact parent directory in monitor.py. It uses only
Python's standard library; no GPU model is loaded by monitoring.

Every60 seconds it records training progress, finite-loss flags, saved complete
states and per-checkpoint inference ledger counts under `PARENT/monitor/`.
`status.json` and `index.html` show the latest state; `events.jsonl` records only
phase/checkpoint changes, errors and completion. A Linux file lock prevents two
watchdogs. `/proc` command identities determine whether target jobs are active.
No progress for30 minutes is flagged; active jobs are never killed, and CPU
checkpoint export is not mistaken for failure just because GPU0 is idle.

If the original queue and all its children have exited, two consecutive checks
confirm the exit. Only when GPU0 is free can the watchdog run a missing stage:

- Completed training is reused. Failed transfer is rerun in a fresh directory.
- Incomplete training is restarted from the base in a fresh directory using the
  exact original configuration, followed by its transfer. This is not checkpoint
  resume; the existing resume implementation does not support arbitrary crashes.
- Each method/phase gets at most two new attempts. Original files are not changed.
- Every recovery executes from the original clean87d6e18 checkout, checks its
  identity and the original dataset/config, and explicitly selects only GPU0.
- GPU1/2/3, SDXL processes, dependencies, scientific parameters and source code are
  untouched. No disk-space query or deletion is performed.

NaN, integrity errors and recognized code exceptions are marked `needs_attention`;
this script cannot diagnose and rewrite arbitrary code. It writes evidence and
keeps monitoring. There is no chat notification or automatic Codex wakeup in this
fallback. The parent agent can inspect these records for further diagnosis.

Recovery locations and launch intent are persisted before spawning a subprocess.
After a watchdog restart, live target processes prevent duplicate work. Recovery
outputs live in `PARENT/recovery/`, and monitor/index.html links the selected
results without editing the original experiment index or statuses.

The watchdog exits after both methods have complete1500-step training with all
three snapshots, and297 outputs each. Final checks verify the paired export
records, complete per-step ledgers and all594 PNG checksums. Safety-filtered images
remain valid recorded outcomes; no image-quality success is implied.
