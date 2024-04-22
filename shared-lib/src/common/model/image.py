from transformers import ResNetForImageClassification, ResNetPreTrainedModel
import segmentation_models_pytorch as smp


def resnet(model_base: str) -> ResNetPreTrainedModel:
    return ResNetForImageClassification.from_pretrained(model_base)


def unet(model_base: str):
    ENCODER = 'efficientnet-b4'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid'

    return smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=2,
        activation=ACTIVATION,
    )
