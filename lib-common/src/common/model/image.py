from transformers import ResNetForImageClassification, ResNetPreTrainedModel
import segmentation_models_pytorch as smp

"""
Collection of model factories for image classification and segmentation models
"""


def resnet(model_base: str, **kwargs) -> ResNetPreTrainedModel:
    return ResNetForImageClassification.from_pretrained(model_base, **kwargs)


def unet(model_base: str, **kwargs):
    return smp.Unet(
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',
        activation='sigmoid',
        **kwargs
    )
