from datasets import load_dataset

def ingest_huggingface_dataset(dataset_name):
    # Load a dataset from Hugging Face Hub
    dataset = load_dataset(dataset_name)
    return dataset
