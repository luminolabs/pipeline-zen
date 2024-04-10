# dataset.py
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset
from datasets import load_dataset
from huggingface_hub.utils._errors import HfHubHTTPError

class AlzheimerMRIDataset(Dataset):
    def __init__(self, split='train'):
        # self.dataset = load_dataset('Falah/Alzheimer_MRI', split=split, local_files_only=True)
        self.dataset = load_dataset('Falah/Alzheimer_MRI', split=split)
        # try:
        #     # First attempt to load the dataset normally
        #     self.dataset = load_dataset(dataset_name, split=split)
        # except HfHubHTTPError as e:
        #     # Check if the error is a 5XX server error
        #     if 500 <= e.response.status_code < 600:
        #         print("Server error encountered. Attempting to load the dataset from local cache.")
        #         try:
        #             # Attempt to load from the local cache
        #             self.dataset = load_dataset(dataset_name, split=split, local_files_only=True)
        #         except FileNotFoundError:
        #             print("Dataset not found in local cache. Please ensure it has been downloaded at least once.")
        #     else:
        #         raise
        print("Transforming dataset!")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image for ResNet-50
            transforms.Grayscale(num_output_channels=3), # Convert grayscale to 3-channel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example['image']
        label = example['label']
        image = self.transform(image)
        return image, label
