import os

from common.utils import is_environment, Env

# Redis broker
broker_redis_host = os.environ.get('REDIS_HOST', '127.0.0.1')
broker_redis_port = os.environ.get('REDIS_PORT', 6379)
broker_redis_db = int(os.environ.get('REDIS_CELERY_DB', 0))
broker_redis_url = f'redis://{broker_redis_host}:{broker_redis_port}/{broker_redis_db}'

# Memory broker (for local env)
broker_memory_url = 'memory://localhost/'

# Celery specific app configuration
# see: https://docs.celeryq.dev/en/latest/userguide/configuration.html
if is_environment(Env.LOCAL):
    broker_url = broker_memory_url
else:
    broker_url = broker_redis_url
