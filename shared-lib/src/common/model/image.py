from transformers import ResNetForImageClassification, ResNetPreTrainedModel
import segmentation_models_pytorch as smp


def resnet(model_base: str) -> ResNetPreTrainedModel:
    return ResNetForImageClassification.from_pretrained(model_base)


def unet(model_base: str):
    return smp.Unet(
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',
        classes=2,
        activation='sigmoid',
    )
