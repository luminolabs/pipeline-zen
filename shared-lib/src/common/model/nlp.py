from transformers import AutoModelForSequenceClassification, PreTrainedModel


def auto(model_base: str) -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(model_base)