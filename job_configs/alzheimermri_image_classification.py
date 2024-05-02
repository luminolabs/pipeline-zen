from job_configs._defaults import build_job_config, JobCategory, JobType

_job_config = {
    # Used to associate results and scores in logs and bigquery
    'job_id': 'alzheimermri_image_classification',

    # What kind of job is this?
    'category': JobCategory.IMAGE,
    'type': JobType.CLASSIFICATION,

    # Dataset provider configuration
    'dataset_id': 'Falah/Alzheimer_MRI',
}
job_config = build_job_config(_job_config)
