from transformers import AutoModelForSequenceClassification, PreTrainedModel

from common.model.utils import add_hf_params

"""
Collection of factories for NLP models
"""


def auto(model_base: str, **kwargs) -> PreTrainedModel:
    kwargs = add_hf_params(model_base, **kwargs)
    return AutoModelForSequenceClassification.from_pretrained(model_base, **kwargs)
