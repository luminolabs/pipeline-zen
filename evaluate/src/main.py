import asyncio

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from common.dataset.preprocessor.utils import dataset_preprocess_factory
from common.dataset.provider.utils import dataset_provider_factory
from common.dataset.kind.utils import dataset_kind_factory
from common.model.utils import model_factory


async def main():
    # TODO: This needs to come from external input
    job_config = {
        # Dataset provider configuration
        'dataset_provider': 'huggingface',
        'dataset_id': 'Falah/Alzheimer_MRI',

        # Train / test dataset splits mapping
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
        'shuffle': True,

        # Output configuration
        'model_weights_path': './.results/trained_model.pth',
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

    # TODO: Deduplicate code above, which is the same as in the train workflow

    print("Fetching the model")
    # Instantiate the appropriate model
    model = model_factory(
        model_kind=job_config.get('dataset_kind'),
        model_base=job_config.get('model_base'))
    model.load_state_dict(torch.load(job_config.get('model_weights_path')))
    model.to(device)
    model.eval()

    # TODO: Make code below configurable; most of the code below will be generalized and put into a lib

    # Variables to store predictions and actual labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)

            # Store predictions and actual labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays for metric calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')  # Adjust average as needed
    recall = recall_score(all_labels, all_preds, average='macro')  # Adjust average as needed
    f1 = f1_score(all_labels, all_preds, average='macro')  # Adjust average as needed

    # Print the metrics
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')


asyncio.run(main())
