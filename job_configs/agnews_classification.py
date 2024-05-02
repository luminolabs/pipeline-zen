job_config = {
    # Used to associate results and scores
    'job_id': 'agnews-classification',

    # Dataset provider configuration
    'dataset_provider': 'huggingface',
    'dataset_id': 'ag_news',

    # Train / test dataset splits mapping
    'train_split': 'train',
    'test_split': 'test',

    # Dataset configuration
    'dataset_kind': 'single_label',
    'single_label_dataset_config': {
        'input_col': 'text',
        'label_col': 'label',
    },

    # Data preprocessing configuration
    'preprocessor': 'text_transforms',
    'text_transforms_dataset_config': {
        'transforms_input_func': 'strip',
        'transforms_label_func': None,
    },

    # Tokenizer configuration
    'tokenizer_id': 'google-bert/bert-base-cased',

    # Model configuration
    'model_base': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'model_base_args': {
        'ignore_mismatched_sizes': True,
        'id2label': {
            "0": "World",
            "1": "Sports",
            "2": "Business",
            "3": "Sci/Tech"
        },
        'label2id': {
            'World': 0,
            'Sports': 1,
            'Business': 2,
            'Sci/Tech': 3,
        }
    },

    # Training configuration
    'batch_size': 18,  # ~13.5GB memory use
    'num_epochs': 5,
    'learning_rate': 0.0005,
    'shuffle': False,
    # On every epoch, stop after this number of batches
    'num_batches': None,  # ex 5
    # Loss function configuration
    'loss_func_name': 'CrossEntropyLoss',
    'loss_func_args': {},
}
