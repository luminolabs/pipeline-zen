import torch
from torch.utils.data import DataLoader
from dataset import AlzheimerMRIDataset # Adjust this import based on your dataset
from model import get_model # Adjust this import to load your model architecture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def evaluate_model():
    # Parameters
    num_classes = 4  # Update this based on your dataset specifics
    batch_size = 32

    # Prepare the evaluation dataset and DataLoader
    eval_dataset = AlzheimerMRIDataset(split='test')  # Adjust the dataset loading as per your requirements
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = get_model(num_classes)  # Load your model architecture
    model_path = 'trained_model.pth'  # Path to your saved model
    model.load_state_dict(torch.load(model_path))
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Variables to store predictions and actual labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)
            
            # Store predictions and actual labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays for metric calculation
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')  # Adjust average as needed
    recall = recall_score(all_labels, all_preds, average='macro')  # Adjust average as needed
    f1 = f1_score(all_labels, all_preds, average='macro')  # Adjust average as needed

    # Print the metrics
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

if __name__ == "__main__":
    evaluate_model()
