# SD2.1 Texture context-routing diagnostic

Exactly18 frozen-weight images: three conditions × street/snow/flowers × seeds31/38. All use the **original** LoRA Mailbox S1000, orange C150 and wood T400 paired adapters/embeddings. This does not use either background-preservation continuation. There is no training, new parameter, strength sweep or change to PNDM100/CFG3.5/512px/fp16/safety-checker sampling. Only GPU0 is visible.

| Condition | Human prompt | Conditioning intervention |
|---|---|---|
| `baseline_sct` | `a photo of <S*> mailbox in <C*> color with <T*> texture ` + scene suffix | Original SCT hidden and residual gates. |
| `true_SC` | `a photo of <S*> mailbox in <C*> color ` + scene suffix | Removes the whole Texture phrase and sets the Texture residual gate to zero. |
| `t_residual_initializer` (Q1) | Same prompt as baseline | The Texture K/V residuals alone receive auxiliary CLIP hidden. Base K/V and Subject/Color residuals keep the original SCT hidden. |

True SC is a practical two-branch control: deleting Texture semantics also changes the later token positions and contextual hidden states, while its residual is disabled. It is **not** a one-variable Texture-residual ablation. The pipeline retains the same loaded token table and processor slots to avoid unrelated token-ID/loading changes, but the Texture token is absent from the SC prompt and both CFG rows of its gate are zero.

For Q1, auxiliary77-token IDs are copied from the baseline encoding. Only the learned S and C IDs are replaced, at their existing positions, by the ordinary single-token initializer IDs for `mailbox` and `orange`. The learned T ID, all other tokens, sequence length and positional indices stay unchanged. A multi-token initializer aborts this experiment; no BPE piece is silently selected. This deliberately creates the auxiliary sequence corresponding to `... mailbox mailbox in orange color with <T*> texture ...`. It is not shown to the user as the generation prompt. The full auxiliary hidden sequence is passed to attention, but only its Texture-gated residual output is used.

The CFG unconditional auxiliary row is copied from the unchanged main empty-prompt hidden. All residual gates on this row remain zero. Main direct CLIP hidden is checked against the actual pipeline `_encode_prompt` path; the auxiliary path follows the same attention-mask setting and dtype. The current SD2.1 topology must have no cross-attention context normalization. Audit output records main/aux IDs, gate masks, initializer IDs, hidden hashes and the T hidden difference. This tests one contextual input route; it does not remove learned S/C effects from base K/V or from every ordinary contextual token, and cannot prove complete branch independence.

The entrypoint reuses the prompt-order experiment's live source audit and the original96-image screen. It first reproduces all six original T400 SCT images; **all image SHA values and safety states must match before either control can generate**. The same installed routing processor is used during this check, validating its no-auxiliary path. A read-only hook records the actual first UNet input after scheduler scaling and requires equality across every condition/scene with the same seed. Every image receives a fresh device generator. Source locks and checkout identity are checked again after generation. Filtering is recorded, never silently rerun.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python -u -B -m experiments.sd21_context_routing_v1.run \
  --screen-audit /ABSOLUTE/SCREEN96/experiment/source_audit.json \
  --output /ABSOLUTE/FRESH_EXTERNAL_OUTPUT
```

The output contains the18-row manifest, per-image ledger, source audit/selection, `text_conditioning_audit.json`, actual latent hashes, status and a three-column HTML comparison. Execution success is not a quality conclusion. CPU protocol tests: `python -B -m unittest discover -s experiments/sd21_context_routing_v1 -p test_protocol.py`. Attention tensor tests in the same directory require the actual torch environment.
