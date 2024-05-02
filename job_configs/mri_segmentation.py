job_config = {
    # Used to associate results and scores
    'job_id': 'mri-segmentation',

    # Dataset provider configuration
    'dataset_provider': 'huggingface',
    'dataset_id': 'rainerberger/Mri_segmentation',
    'dataset_fetch_config': {
        # If there's a subset name, set it here
        'name': None,
    },

    # Train / test dataset splits mapping
    'train_split': 'train',
    'test_split': 'test',

    # Dataset configuration
    'dataset_kind': 'single_label',
    'single_label_dataset_config': {
        'input_col': 'image',
        'label_col': 'annotation',
    },

    # Data preprocessing configuration
    'preprocessor': 'torchvision_transforms',
    'torchvision_transforms_dataset_config': {
        'transforms_input_func': 'to_tensor',
        'transforms_label_func': 'to_tensor',
    },

    # Tokenizer configuration
    'tokenizer_id': None,

    # Model configuration
    'model_base': 'unet',
    'model_base_args': {
        'classes': 1,
    },

    # Training configuration
    'batch_size': 8,
    'num_epochs': 10,
    'learning_rate': 0.001,
    'shuffle': False,
    # On every epoch, stop after this number of batches
    'num_batches': None,  # ex 5
    # Loss function configuration
    'loss_func_name': 'FocalLoss',
    'loss_func_args': {
        'mode': 'binary',
    },
}
