from job_configs._defaults import build_job_config, JobCategory, JobType

_job_config = {
    # Used to associate results and scores
    'job_id': 'imdb_nlp_classification',

    # What kind of job is this?
    'category': JobCategory.NLP,
    'type': JobType.CLASSIFICATION,

    # Dataset provider configuration
    'dataset_id': 'stanfordnlp/imdb',

    # Tokenizer configuration
    'tokenizer_id': 'google-bert/bert-base-cased',
}
job_config = build_job_config(_job_config)
