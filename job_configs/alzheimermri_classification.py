job_config = {
    # Used to associate results and scores
    'job_id': 'alzheimermri-classification',

    # Dataset provider configuration
    'dataset_provider': 'huggingface',
    'dataset_id': 'Falah/Alzheimer_MRI',

    # Train / test dataset splits mapping
    'train_split': 'train',
    'test_split': 'test',

    # Dataset configuration
    'dataset_kind': 'single_label',
    'single_label_dataset_config': {
        'input_col': 'image',
        'label_col': 'label',
    },

    # Data preprocessing configuration
    'preprocessor': 'torchvision_transforms',
    'torchvision_transforms_dataset_config': {
        'transforms_input_func': 'transforms_set_1',
        'transforms_label_func': None,  # Not applicable for classification
    },

    # Tokenizer configuration
    'tokenizer_id': None,

    # Model configuration
    'model_base': 'microsoft/resnet-50',

    # Training configuration
    'batch_size': 16,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'shuffle': False,
    # On every epoch, stop after this number of batches
    'num_batches': None,  # ex 5
    # Loss function configuration
    'loss_func_name': 'CrossEntropyLoss',
    'loss_func_args': {},
}
