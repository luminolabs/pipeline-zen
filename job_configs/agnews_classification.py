from job_configs._defaults import build_job_config, JobCategory, JobType

_job_config = {
    # Used to associate results and scores
    'job_id': 'agnews-classification',

    # What kind of job is this?
    'category': JobCategory.NLP,
    'type': JobType.CLASSIFICATION,

    # Dataset provider configuration
    'dataset_id': 'ag_news',

    # Model configuration
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
}
job_config = build_job_config(_job_config)
