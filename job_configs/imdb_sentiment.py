job_config = {
    # Used to associate results and metrics
    'job_id': 'imdb-sentiment',

    # Dataset provider configuration
    'dataset_provider': 'huggingface',
    'dataset_id': 'stanfordnlp/imdb',

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
        'transforms_input_func': 'transforms_set_1'
    },

    # Tokenizer configuration
    'tokenizer_id': 'google-bert/bert-base-cased',

    # Model configuration
    'model_base': 'cardiffnlp/twitter-roberta-base-sentiment-latest',

    # Training configuration
    'batch_size': 42,
    'num_epochs': 2,
    'learning_rate': 0.00001,
    'shuffle': False,
    # On every epoch, stop after this number of batches
    'num_batches': None,  # ex 5
    # Loss function configuration
    'loss_func_name': 'cross_entropy',
    'loss_func_args': {},
}
