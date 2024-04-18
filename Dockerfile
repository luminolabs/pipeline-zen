FROM python

# Install essentials
RUN apt-get update \
	&& apt-get install -y \
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

# Install python libraries needed by the shared-lib
COPY shared-lib/requirements.txt ./requirements-shared.txt
RUN pip install -r requirements-shared.txt

ARG TARGET_WORKFLOW

# Install python libraries needed by the workflow
COPY ${TARGET_WORKFLOW}/requirements.txt .
RUN pip install -r requirements.txt

# Copy shared-lib source code
COPY shared-lib/src .

# Copy workflow source code
COPY ${TARGET_WORKFLOW}/src .

# Copy job configurations
COPY job-configs .

# Set environment to `docker`
# This affects a few runtime options such as cache and results folders
ENV ENVIRONMENT=docker

# Run workflow
ENTRYPOINT ["python", "main.py"]

# NOTE: `.cache` and `.results` folders should be mounted with the `docker run` command, see readme