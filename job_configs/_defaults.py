from enum import Enum

from common.utils import JsonEnumBase

DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_EPOCHS = 10
DEFAULT_NUM_BATCHES = None  # `None` for no limit


class JobCategory(JsonEnumBase):
    NLP = 'nlp'
    IMAGE = 'image'
    LLM = 'llm'


class JobType(JsonEnumBase):
    CLASSIFICATION = 'classification'
    SEGMENTATION = 'segmentation'
    TEXT_GENERATION = 'text_generation'


def build_job_config(job_config: dict) -> dict:
    """
    Mix the input configuration with the preset configurations defined below

    :param job_config: The job config overrides
    :return: The final job config dictionary
    """
    def make_key(category: JobCategory, type: JobType) -> str:
        return f'{category}_{type}'

    # Match the input job config with one of the presets that are defined below
    key = make_key(job_config['category'], job_config['type'])
    if key == make_key(JobCategory.NLP, JobType.CLASSIFICATION):
        return {**nlp_classification_base, **job_config}
    elif key == make_key(JobCategory.IMAGE, JobType.CLASSIFICATION):
        return {**image_classification_base, **job_config}
    elif key == make_key(JobCategory.IMAGE, JobType.SEGMENTATION):
        return {**image_segmentation_base, **job_config}


job_config_base = {
    # Used to associate results and scores in logs and bigquery
    'job_id': None,

    # What kind of job is this?
    'category': None,
    'type': None,

    # Dataset provider configuration
    'dataset_provider': 'huggingface',
    'dataset_id': None,
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
        'input_col': None,
        'label_col': None,
        'master_column': None,
    },

    # Data preprocessing configuration
    'preprocessor': None,
    '<preprocessor name>_dataset_config': {
        'transforms_input_func': None,
        'transforms_label_func': None,
    },

    # Tokenizer configuration
    'tokenizer_id': None,

    # Model configuration
    'model_base': None,
    'model_base_args': {},

    # Training configuration
    'batch_size': DEFAULT_BATCH_SIZE,
    'num_epochs': DEFAULT_NUM_EPOCHS,
    'learning_rate': 0.001,
    'shuffle': True,
    # On every epoch, stop after this number of batches
    'num_batches': DEFAULT_NUM_BATCHES,  # ex 5
    # Loss function configuration
    'loss_func_name': 'CrossEntropyLoss',
    'loss_func_args': {},
}
nlp_classification_base = {**job_config_base, **{
    # Dataset configuration
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
}}
image_classification_base = {**job_config_base, **{
    # Dataset configuration
    'single_label_dataset_config': {
        'input_col': 'image',
        'label_col': 'label',
    },

    # Data preprocessing configuration
    'preprocessor': 'torchvision_transforms',
    'torchvision_transforms_dataset_config': {
        'transforms_input_func': 'transforms_set_1',
        'transforms_label_func': None,
    },

    # Model configuration
    'model_base': 'microsoft/resnet-50',
}}
image_segmentation_base = {**job_config_base, **{
    # Dataset configuration
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

    # Model configuration
    'model_base': 'unet',

    # Loss function configuration
    'loss_func_name': 'FocalLoss',
    'loss_func_args': {
        'mode': 'binary',
    },
}}
