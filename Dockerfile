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

ARG TARGET_WORKFLOW

# Install python libraries needed by the workflow
COPY lib-workflows/${TARGET_WORKFLOW}/requirements.txt .
RUN pip install -r requirements.txt

# Copy application configuration folder
COPY app-configs app-configs

# Copy lib-common source code
COPY lib-common/src .

# Copy workflow source code
COPY lib-workflows/${TARGET_WORKFLOW}/src .

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
WORKDIR /project/${TARGET_WORKFLOW}

# Run workflow
ENTRYPOINT ["python", "cli.py"]
