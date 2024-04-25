from torchvision.transforms import transforms

"""
Collection of preprocessing functions for torchvision datasets.
"""


transforms_set_1 = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image for ResNet-50
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


to_tensor = transforms.Compose([
        transforms.ToTensor(),
    ])
