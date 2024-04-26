from transformers import ResNetForImageClassification, ResNetPreTrainedModel
import segmentation_models_pytorch as smp

"""
Collection of model factories for image classification and segmentation models
"""


def resnet(model_base: str) -> ResNetPreTrainedModel:
    return ResNetForImageClassification.from_pretrained(model_base)


def unet(model_base: str, num_classes: int):
    return smp.Unet(
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',
        classes=num_classes,
        activation='sigmoid',
    )
