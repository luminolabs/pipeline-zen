from common.config_manager import config

# Celery configuration
# see: https://docs.celeryq.dev/en/latest/userguide/configuration.html
broker_url = config.celery_broker_url
