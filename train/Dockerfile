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

# Install rest of python libraries
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY src .

# Run workflow
ENTRYPOINT ["python", "main.py"]

# TODO: Add volume for dataset `.cache`, and results `.results` dirs,
# so that dirs stay persistent between runs