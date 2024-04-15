import asyncio
import os
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset.preprocessor.utils import dataset_preprocess_factory
from dataset.provider.utils import dataset_provider_factory
from dataset.kind.utils import dataset_kind_factory
from model.utils import model_factory


async def main():
    # TODO: This needs to come from external input
    job_config = {
        # Dataset provider configuration
        'dataset_provider': 'huggingface',
        'dataset_id': 'Falah/Alzheimer_MRI',

        # Train / test dataset split mapping
        'train_split': 'train',
        'test_split': 'test',

        # Dataset configuration
        'dataset_kind': 'image',
        'image_dataset_config': {
            'image_col': 'image',
            'label_col': 'label',
        },

        # Data preprocessing configuration
        'preprocessor': 'torchvision_transforms',
        'torchvision_transforms_dataset_config': {
            'transforms_func': 'transforms_set_1',
        },

        # Model configuration
        'model_base': 'microsoft/resnet-50',

        # Training configuration
        'num_classes': 4,
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'shuffle': True
    }

    print("Loading and configuring dataset!")
    # This is the dataset that pulls from the content provider
    # ex. huggingface, s3 providers
    dataset = dataset_provider_factory(
        dataset_provider=job_config.get('dataset_provider'),
        dataset_id=job_config.get('dataset_id'),
        split=job_config.get('train_split'))
    dataset = await dataset.fetch()
    # This is the dataset that's specialized for the type of data we're looking to train
    # ex. image data
    dataset_kind = dataset_kind_factory(
        dataset_kind=job_config.get('dataset_kind'),
        dataset=dataset,
        **job_config.get(job_config.get('dataset_kind') + '_dataset_config'))
    # This is the preprocessing dataset,
    # it will apply transformations and prepare data for training
    # ex. torchvision transforms
    dataset_preprocess = dataset_preprocess_factory(
        dataset_preprocess=job_config.get('preprocessor'),
        dataset=dataset_kind,
        **job_config.get(job_config.get('preprocessor') + '_dataset_config'))

    # This loads data from the dataset in batches;
    # data requested from the dataloader will return preprocessed
    dataloader = DataLoader(
        dataset_preprocess,
        batch_size=job_config.get('batch_size'),
        shuffle=job_config.get('shuffle'))

    # To run on GPU or not to run on GPU, that is the question
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on (CPU/GPU?) device:", device)

    print("Fetching the model")
    # Instantiate the appropriate model
    model = model_factory(
        model_kind=job_config.get('dataset_kind'),
        model_base=job_config.get('model_base'))
    model.to(device)

    # TODO: Make code below configurable; most of the code below will be moved to the `training` package
    # TODO: Implement metrics lib, to capture timing, model, etc metrics
    # TODO: Use logger instead of print

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=job_config.get('learning_rate'))
    print("Loss and Optimizer is set")

    # Capture the start time
    start_time = datetime.now()
    print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch in range(job_config.get('num_epochs')):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{job_config.get("num_epochs")}, Loss: {running_loss/len(dataloader)}')

    # Capture the end time
    end_time = datetime.now()
    print(f"Training ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate and print the total training time
    total_time = end_time - start_time
    total_minutes = total_time.total_seconds() / 60
    print(f"Total training time: {total_minutes:.2f} minutes")

    # TODO: Implement different storage strategies; ex. gcp/s3 bucket
    print("Training loop complete, now saving the model")
    # Save the trained model
    os.makedirs('.results', exist_ok=True)
    model_path = os.path.join('.results', 'trained_model.pth')
    torch.save(model.state_dict(), model_path)
    print("Trained model saved!")


asyncio.run(main())


