from transformers import PreTrainedModel

from common.model import image, nlp


def model_factory(model_kind: str, model_base: str, **kwargs) -> PreTrainedModel:
    print(f'Using `{model_kind}.{model_base}` model')
    if model_kind == 'input_label':
        if 'resnet' in model_base:
            return image.resnet(model_base)
        elif 'sentiment' in model_base:
            return nlp.auto(model_base)
        elif 'unet' == model_base:
            return image.unet(model_base, **kwargs)
    if model_kind == '...':
        pass
