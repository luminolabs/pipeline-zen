# model.py
from transformers import ResNetForImageClassification
import torch.nn as nn

def get_model(num_classes):
    # Load the pretrained ResNet-50 model
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    #print(type(model))
    # Replace the final layer with a new one adapted to your number of classes
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)
    return model
