# Server preflight and runtime blocker

Date: 2026-09-01 (Asia/Shanghai)

## Completed

- Local implementation commit: `eb454dd4d2fad4018f6251f5f7891200a4d5bd0d`.
- GitHub branch: `repro/2026-09-01-colorpeel-two-object`.
- Server checkout was clean before deployment.
- Server checkout was moved to the new branch without reset or destructive file
  operations and verified at the same commit.
- `git pull --ff-only` reported `Already up to date`.
- Full server test suite in `colorpeel017`: `149 passed in 26.62s`.

## Blocker

The SSH connection was reset immediately after the test suite. A new login was
rejected by the server with:

```text
Your account has expired; please contact your system administrator.
```

No render plan or Blender runtime smoke was started. No training was started.
The next safe action is to restore the same server account, re-check the
checkout and GPU 3, generate the 360-request plan in a new external run
directory, then run the two-request real Blender smoke.
