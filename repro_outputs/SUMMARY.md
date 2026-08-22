# ColorPeel-on-CLEVR reproduction summary

- GitHub fork: `https://github.com/Joeysunnn/color-peel-ice.git`
- Official upstream baseline: `021f5c74cee6c231a03b8b49bb96750cadfc4e06`
- Working branch: `repro/2026-08-21-colorpeel-clevr`
- Server-verified handoff commit: `e6c57d1ba9074db50f07a32cb56bebaffcc44876`
- Overall status: `partial`
- Current code/test status: `success` (`44 passed` in the isolated pytest suite)
- Training/evaluation status: `not_run`

The 3×3 CLEVR adaptation, six-token gradient-boundary correction, data adapter,
auditable run launcher, deterministic 900-item generation protocol, scorers,
project structure, experiment registry, and literature records were pushed in
the earlier implementation commit. The newer dual-smoke observation contract
and independent Grounded-SAM/Qwen stage changes are included in this pre-run
handoff and passed the current isolated 44-test suite.

The server checkout at `/home/r12user5/Documents/Jiawei/colorpeel/` was created
directly from that GitHub branch and is currently verified clean at commit
`e6c57d1`. The newer pre-run handoff changes have not yet been fast-forwarded
to the server. Earlier server-side setup from the superseded workflow was
removed: the isolated
`colorpeel017` environment and temporary adapter directory no longer exist.
Shared caches and all pre-existing files were left untouched.

No model download, smoke training, 1500-step training, image generation,
Grounded-SAM segmentation, Qwen3-VL prediction, or experiment metric run is
claimed. Literal official AdamW decay is now the locked policy: ordinary
vocabulary embedding drift will be observed and reported, not restored and not
treated as a run failure by itself.

Next safe action: recreate the environment from the GitHub checkout, run the
tracked data audit and launcher preflights, then execute the independent
two-step and nine-step real training smokes with token-observation outputs.
