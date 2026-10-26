# Completed native mailbox material test

Execution commit: `9c6c422`; no training or learned adapters/tokens.

Server result:
`/var/tmp/colorpeel-r12user5/runs/sd15_native_mailbox_material_v1/20261026-232332__native10__9c6c422__42`

All10 images completed;0 safety-filtered. Each512px PNG matches its recorded SHA256.
Runtime confirms stock `diffusers.models.unet_2d_condition.UNet2DConditionModel`,
stock `diffusers.models.attention_processor.AttnProcessor`, and float16 weights.
The frozen SD1.5 source receipt passed before/after verification. The shared
SD1.4 safety checker remains enabled, matching the existing Subject tests.

Five columns: metallic, matte, glossy plastic, ceramic, wood. Two rows:seeds42/43.
Prompt: `a photo of a mailbox with a {material} surface`. PNDM100,CFG3.5,512px,
fresh CUDA generator per image; no custom attention mask or reference image.

## Visual observations

- Seed42 retains a pale wall-mounted mailbox across material words. `wood` changes
  the background to wooden planks while the mailbox front remains pale/painted-looking.
- Seed43 retains silver/metal-looking fronts across all five prompts. `wood` alters
  the side/post appearance without clearly converting the whole mailbox to wood.
- Metallic/matte show some highlight differences, but glossy-plastic/ceramic do not
  consistently produce clearly distinct surface appearances across both seeds.
- Generated lettering and mailbox geometry vary. These generic mailboxes are not
  reconstructions of the original orange photograph.

Inference: for this exact wording and low-CFG recipe, the base model's material
binding is already unreliable and may affect surrounding objects/background.
This motivates checking the base-model prompt response when interpreting later
Subject-token experiments. It does not isolate the cause of any learned failure.
Only two seeds were tested; no physical BRDF ground truth or quantitative material
accuracy was measured. The result does not establish inability under other prompts
or sampling settings.

`contact_sheet.png`, all10 PNGs, prompt/status ledgers, actual runtime configuration
and `repro_outputs/` are saved in the result directory. Previous results are unchanged.
