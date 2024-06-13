from logging import Logger
from typing import Optional

from transformers import PreTrainedModel

from common.model import image, nlp, llm


def model_factory(model_kind: Optional[str], model_base: str, logger: Logger, **kwargs) -> PreTrainedModel:
    """
    Factory method for instantiating a pre-trained model.

    :param model_kind: Kind of model to instantiate
    :param model_base: Base model name
    :param logger: Logger instance
    :param kwargs: Keyword arguments to pass to the model
    :return: Pretrained model instance
    """
    logger.info(f'Using `{model_kind}.{model_base}` model')
    if model_kind == 'single_label':
        if 'resnet' in model_base:
            return image.resnet(model_base, **kwargs)
        elif any(x in model_base for x in ('bert', 't5',)):
            return nlp.auto(model_base, **kwargs)
        elif 'unet' == model_base:
            return image.unet(model_base, **kwargs)
        else:
            raise TypeError(f'model_base: {model_base} is not a valid option '
                            f'for model_kind: {model_kind}')

    # For torchtune configurations
    if not model_kind:
        if any(x in model_base for x in ('llama',)):
            return llm.auto(model_base, **kwargs)

    if model_kind == '...':
        pass
    else:
        raise TypeError(f'model_kind: {model_kind} is not a valid option')
