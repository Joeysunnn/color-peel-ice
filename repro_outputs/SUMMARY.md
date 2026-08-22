# ColorPeel-on-CLEVR reproduction summary

- GitHub fork: `https://github.com/Joeysunnn/color-peel-ice.git`
- Official upstream baseline: `021f5c74cee6c231a03b8b49bb96750cadfc4e06`
- Working branch: `repro/2026-08-21-colorpeel-clevr`
- Verified implementation commit: `41d752a9d8e8b3a5ab711db90990ab28e4f58000`
- Overall status: `partial`
- Code/test status: `success` (local static and unit tests)
- Training/evaluation status: `not_run`

The 3×3 CLEVR adaptation, six-token gradient-boundary correction, data adapter,
auditable run launcher, deterministic 900-item generation protocol, scorers,
project structure, experiment registry, and literature records are implemented
locally and pushed to the fork.

The server checkout at `/home/r12user5/Documents/Jiawei/colorpeel/` was created
directly from that GitHub branch and verified clean at commit `41d752a`. Earlier
server-side setup from the superseded workflow was removed: the isolated
`colorpeel017` environment and temporary adapter directory no longer exist.
Shared caches and all pre-existing files were left untouched.

No model download, smoke training, 1500-step training, image generation,
Grounded-SAM segmentation, Qwen3-VL prediction, or experiment metric run is
claimed. The AdamW weight-decay effect on non-modifier embedding rows remains a
human decision point and blocks training.

Next safe action: choose whether to preserve literal official AdamW behavior or
add a second value-restoration patch, then create the environment from the
GitHub checkout and run the tracked data audit plus launcher dry-run.
