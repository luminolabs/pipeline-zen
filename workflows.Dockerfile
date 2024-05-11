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
COPY lib-workflows/${TARGET_WORKFLOW}/requirements.txt requirements-workflow.txt
RUN pip install -r requirements-workflow.txt

# Copy application configuration folder
COPY app-configs app-configs

# Copy job configuration folder
COPY job-configs job-configs

# Copy lib-common source code
COPY lib-common/src .

# Copy workflow source code
COPY lib-workflows/${TARGET_WORKFLOW}/src .

# Python libraries are copied to `/project`, include them in the path
ENV PYTHONPATH=/project

# Run workflow
ENTRYPOINT ["python", "${TARGET_WORKFLOW}/cli.py"]
