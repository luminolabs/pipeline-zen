import os

# App configuration
redis_host = os.environ.get('REDIS_HOST', '127.0.0.1')
redis_port = os.environ.get('REDIS_PORT', 6379)
redis_celery_db = int(os.environ.get('REDIS_CELERY_DB', 0))


# Celery specific app configuration
# see: https://docs.celeryq.dev/en/latest/userguide/configuration.html
broker_url = f'redis://{redis_host}:{redis_port}/{redis_celery_db}'
