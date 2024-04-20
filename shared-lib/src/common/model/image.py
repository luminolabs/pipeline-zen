from transformers import ResNetForImageClassification, ResNetPreTrainedModel


def resnet(model_base: str) -> ResNetPreTrainedModel:
    return ResNetForImageClassification.from_pretrained(model_base)
