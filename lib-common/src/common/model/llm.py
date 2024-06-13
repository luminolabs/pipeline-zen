import huggingface_hub
from transformers import AutoModelForCausalLM, PreTrainedModel

"""
Collection of factories for LLM models
"""


def auto(model_base: str, **kwargs) -> PreTrainedModel:
    huggingface_hub.snapshot_download(model_base, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_base, **kwargs)
