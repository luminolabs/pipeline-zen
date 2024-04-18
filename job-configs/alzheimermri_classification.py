job_config_ = job_config = {
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
