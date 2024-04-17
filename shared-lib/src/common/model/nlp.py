from transformers import AutoModelForSequenceClassification


def auto(model_base: str):
    return AutoModelForSequenceClassification.from_pretrained(model_base)