# Three-attribute protocol

The accepted subject+color checkpoints are retained as the two-axis baseline.
The known Fold C red-sphere black upper cap is recorded as a metal-appearance
artifact and does not block this stage. The material gate specifically tests
whether the artifact transfers incorrectly to red rubber spheres.

ColorPeel's CAA implementation and total weight remain unchanged. With three
modifier tokens in every prompt, the existing function averages the three
pairwise distances (subject-color, subject-material, color-material). Thus the
total CAA coefficient remains 0.2 while each pair contributes through that
three-pair mean; this is an explicit protocol consequence, not a new loss.

The checkpoint-1000 accelerator state is saved as evidence only. The upstream
training entry does not implement resume loading, so this stage does not claim
that training recovery has been validated.
