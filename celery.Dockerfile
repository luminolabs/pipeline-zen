FROM python

# Install essentials
RUN apt update \
	&& apt install -y \
		build-essential \
		ca-certificates \
		curl \
		git \
		libssl-dev \
		software-properties-common

# Upgrade pip
RUN python -m pip install --upgrade pip

# Work in this folder
WORKDIR /project

# Install these python libs outside of requirements.txt since they are large libraries
# and we don't want them to be build every time we add a new entry in requirements.txt
RUN pip install torch torchvision transformers

# Install python libraries needed by the lib-common
COPY lib-common/requirements.txt ./requirements-lib-common.txt
RUN pip install -r requirements-lib-common.txt

# Install python libraries needed by the lib-celery
COPY lib-celery/requirements.txt ./requirements-lib-celery.txt
RUN pip install -r requirements-lib-celery.txt

# Install python libraries needed by the workflows
COPY lib-workflows/train/requirements.txt requirements-train.txt
RUN pip install -r requirements-train.txt
COPY lib-workflows/evaluate/requirements.txt requirements-evaluate.txt
RUN pip install -r requirements-evaluate.txt

# Copy lib-common source code
COPY lib-common/src .

# Copy lib-celery source code
COPY lib-celery/src .

# Copy workflow source code
COPY lib-workflows/train/src .
COPY lib-workflows/evaluate/src .

# Copy application configuration folder
COPY app-configs app-configs

# Set GCP credentials file location;
# these are mounted on the container at run time,
# they aren't bundled in the image
ENV GOOGLE_APPLICATION_CREDENTIALS=/project/.secrets/gcp_key.json

# Python libraries are copied to `/project`, include them in the path
ENV PYTHONPATH=/project
# Set the application root path
ENV PZ_ROOT_PATH=/project
# Set the application configuration path
ENV PZ_CONF_PATH=/project/app-configs

# Run workflow from workflow folder
WORKDIR /project/pipeline

# Run workflow
ENTRYPOINT ["python", "train_evaluate.py"]
