from common.model import image, nlp


def model_factory(model_kind: str, model_base: str):
    if model_kind == 'value_label':
        if 'resnet' in model_base:
            return image.resnet(model_base)
        elif 'sentiment' in model_base:
            return nlp.auto(model_base)
    if model_kind == '...':
        # TODO: Implement model factory for other kind of model training, ex. LLM
        pass
