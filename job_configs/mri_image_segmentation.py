from job_configs._defaults import build_job_config, JobType, JobCategory

_job_config = {
    # Used to associate results and scores
    'job_id': 'mri_image_segmentation',

    # What kind of job is this?
    'category': JobCategory.IMAGE,
    'type': JobType.SEGMENTATION,

    # Dataset provider configuration
    'dataset_id': 'rainerberger/Mri_segmentation',

    # Model configuration
    'model_base_args': {
        'classes': 1,
    },
}
job_config = build_job_config(_job_config)
