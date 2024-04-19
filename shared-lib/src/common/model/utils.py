from transformers import PreTrainedModel

from common.model import image, nlp


def model_factory(model_kind: str, model_base: str) -> PreTrainedModel:
    if model_kind == 'input_label':
        if 'resnet' in model_base:
            return image.resnet(model_base)
        elif 'sentiment' in model_base:
            return nlp.auto(model_base)
    if model_kind == '...':
        pass
