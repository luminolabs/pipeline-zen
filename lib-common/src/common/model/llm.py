from transformers import AutoModelForCausalLM, PreTrainedModel

"""
Collection of factories for LLM models
"""


def auto(model_base: str, **kwargs) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(model_base, **kwargs)
