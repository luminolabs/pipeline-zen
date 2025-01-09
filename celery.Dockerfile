# Using older python for torch compatibility,
# especially for multi-GPU training
FROM python:3.10-bullseye

# Use to reset the cache
RUN echo "1"

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
RUN pip install torch==2.4.1 transformers datasets

# Install python libraries needed by the lib-common
COPY lib-common/requirements.txt ./requirements-lib-common.txt
RUN pip install -r requirements-lib-common.txt

# Install python libraries needed by the lib-celery
COPY lib-celery/requirements.txt ./requirements-lib-celery.txt
RUN pip install -r requirements-lib-celery.txt

# Install python libraries needed by the workflows
COPY lib-workflows/torchtunewrapper/requirements.txt requirements-torchtunewrapper.txt
RUN pip install -r requirements-torchtunewrapper.txt

# Copy lib-common source code
COPY lib-common/src .

# Copy lib-celery source code
COPY lib-celery/src .

# Copy workflow source code
COPY lib-workflows/torchtunewrapper/src .

# Copy application configuration folder
COPY app-configs app-configs

# Copy job configuration folder
COPY job-configs job-configs

# Copy VERSION file for record keeping
COPY VERSION .

# Python libraries are copied to `/project`, include them in the path
ENV PYTHONPATH=/project