# TokenVerse-inspired isolation trial: recorded, not adopted

Decision on 2026-11-03: the project owner requested that this adaptation be set aside while working on single-image Qwen initialization. Keep its source, training artifacts and results intact. The next experiment uses no concept-isolation loss and does not resume these weights.

Revision: `b8120dfa1c1c16330bb831250f273044df418a49`. This was an adaptation of the paper's joint-image isolation idea to SD2.1 token-local K/V, not a reproduction of TokenVerse's complete architecture.

Six fresh LoRA64 jobs compared ordinary training against right-half frozen-base consistency (weight1): Mailbox Subject1000, orange Color150 and **wood-spoon Texture1000**. The unrelated-image bank contained25 varied generated images. All4300 total updates were finite, ordinary vocabulary drift was0,12 paired exports and112 outputs passed technical checks. Two evaluation seeds were31/38; this is a bounded diagnostic, not a statistical benchmark.

Visual outcome: no reliable improvement of street/flowers/beach backgrounds in final SCT combinations. Snow often appeared as dots without a convincing environment. One midpoint flower example improved the scene but deformed Subject identity; all three checkpoint steps changed together, so it cannot isolate a causal branch. Texture-only mug examples improved in some cases, while vase/sofa/chair cases regressed. Color-only outputs were largely unchanged. The evidence does not justify adopting this configuration or claiming initialization is the sole cause of the background problem.

Server run: `/home/r12user5/Documents/Jiawei/colorpeel-runs/sd21_tokenverse_isolation_v1/20261103-160000__fresh_sct_paired__b8120df__42/experiment`. The160000 path component is a run identifier, not a verified start time.

Local review: `C:/Users/lenovo/.codex/visualizations/2026/11/02/01a19a4e-9ff6-77d1-8cab-1cf360089268/sd21_tokenverse_isolation_20261103/assessment.html`; full gallery `index.html`; evidence `verification.json`, `training_verification.json`, `qualitative_review.json`; selected examples `summary_examples.png`.
