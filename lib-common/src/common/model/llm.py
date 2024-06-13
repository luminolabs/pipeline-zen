import huggingface_hub
from transformers import AutoModelForCausalLM, PreTrainedModel

from common.model.utils import add_hf_params

"""
Collection of factories for LLM models
"""


def auto(model_base: str, **kwargs) -> PreTrainedModel:
    kwargs = add_hf_params(model_base, **kwargs)
    path = huggingface_hub.snapshot_download(model_base, **kwargs)
    return AutoModelForCausalLM.from_pretrained(path)
