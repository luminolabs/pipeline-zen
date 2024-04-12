# Step 1: Pull job configuration using provided job id, or somehow get it from the filesystem, it depends how
# we want to do this with the Protocol.
# Step 2: Run pipeline (data ingestion, preprocessing, train)
# Step 3: Upload results to storage (ie. some cloud bucket)
# ...
from dataset.utils import dataset_factory

# Fetch job config
# ...get this from args...
job_config = {}

# Pull dataset and convert to torch dataset
dataset = dataset_factory(strategy=job_config.get('dataset_store'), *job_config.get('dataset_store_args'))
torch_dataset = dataset.to_torch_dataset(split='train')

# ... next steps
