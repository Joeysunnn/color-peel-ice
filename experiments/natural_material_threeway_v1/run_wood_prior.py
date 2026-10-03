"""Run the existing Scheme 3 VLM procedure with wood as a surface family."""

from experiments.natural_material_semantic_constraint_v1 import run_semantic_prior as base


base.ALLOWED["surface_family"].add("wood")
base.PROMPT = base.PROMPT.replace(
    "coated_or_painted|bare_metal|plastic|uncertain",
    "coated_or_painted|bare_metal|plastic|wood|uncertain",
)


if __name__ == "__main__":
    base.main()
