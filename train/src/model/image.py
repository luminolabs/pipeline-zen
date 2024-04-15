from transformers import ResNetForImageClassification


def resnet(model_base: str):
    return ResNetForImageClassification.from_pretrained(model_base)