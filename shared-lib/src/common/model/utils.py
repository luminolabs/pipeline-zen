from common.model import image


def model_factory(model_kind: str, model_base: str):
    if model_kind == 'image':
        if 'resnet' in model_base:
            return image.resnet(model_base)
    if model_kind == '...':
        # TODO: Implement model factory for other kind of model training, ex. LLM
        pass
