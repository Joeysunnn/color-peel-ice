# Invalid Subject-Only Training Run

The external run
`20260912-153000__natural_image_subject_color_pilot__subject_recolor_gorilla_short100__cb51b40__42`
completed before a prompt/CAA design error was identified. It is invalid and
must not be used for generation, evaluation, checkpoint composition, or any
subsequent training.

Its staging assigned all five auxiliary-color images the same prompt,
`a photo of <S*>`, and used `cos_weight=0.2` despite having only the single
learned modifier `<S*>`. This incorrectly placed color reconstruction pressure
on the subject token and invoked the single-token CAA fallback.

The replacement protocol binds each image to its corresponding ordinary color
word and requires `cos_weight=0`; the training implementation rejects a
nonzero CAA weight for fewer than two learned modifier tokens.
