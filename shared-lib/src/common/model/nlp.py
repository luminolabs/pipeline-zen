from transformers import AutoModelForSequenceClassification, PreTrainedModel

"""
Collection of factories for NLP models
"""


def auto(model_base: str) -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(model_base)
