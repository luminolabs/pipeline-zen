from typing import List
import argparse
import os

from datasets import load_dataset, DatasetDict, Dataset


def download_dataset_json(hf_ds_name: str, dest: str):
  """Download the given HF dataset to the given (local) dest in JSON format."""
  ds = load_dataset(hf_ds_name)
  datasets_list: List[Dataset] = []
  if type(ds) == DatasetDict:
     for _, dataset in ds.items():
        datasets_list.append(dataset)
  elif type(ds) == Dataset:
     datasets_list.append(ds)

  if not datasets_list:
     raise(f'Unable to find datasets from: {ds} for HF dataset name: {hf_ds_name}')

  num_datasets = len(datasets_list)
  for i, dataset in enumerate(datasets_list):
     dataset_path = os.path.join(dest, f'dataset_part_{i}_of_{num_datasets}.json')
     print(f'Saving dataset {hf_ds_name} to path: {dataset_path}.  Details: ' +
           f'Column names: {dataset.column_names}, Rows: {dataset.num_rows}')
     dataset.to_json(dataset_path)


# Example command-line:  See README
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download HF datasets (See README)')
    parser.add_argument('-hfds', '--hf_dataset_path', type=str, required=True,
                        help='Hugging Face Dataset path (eg: "TIGER-Lab/MathInstruct")')
    parser.add_argument('-dest', '--destination_base_path', type=str, required=False,
                        default='/tmp/', help='Base path of destination directory')
    args = parser.parse_args()
    download_dataset_json(args.hf_dataset_path, args.destination_base_path)
