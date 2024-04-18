job_config = {
    # Used to associate results and metrics
    'job_id': 'alzheimer-mri',

    # Dataset provider configuration
    'dataset_provider': 'huggingface',
    'dataset_id': 'Falah/Alzheimer_MRI',

    # Train / test dataset splits mapping
    'train_split': 'train',
    'test_split': 'test',

    # Dataset configuration
    'dataset_kind': 'input_label',
    'input_label_dataset_config': {
        'input_col': 'image',
        'label_col': 'label',
    },

    # Data preprocessing configuration
    'preprocessor': 'torchvision_transforms',
    'torchvision_transforms_dataset_config': {
        'transforms_func': 'transforms_set_1',
    },

    # Tokenizer configuration
    'tokenizer_id': None,

    # Model configuration
    'model_base': 'microsoft/resnet-50',

    # Training configuration
    'num_classes': 4,
    'batch_size': 32,
    'num_epochs': 2,
    'learning_rate': 0.001,
    'shuffle': False,
    'num_batches': 5,

    # Evaluation configuration
    'model_weights_path': 'alzheimer-mri/2024-04-18-16-12-07.pt'
}
