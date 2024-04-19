import asyncio
import sys

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from common.utils import configure_model_and_dataloader, load_job_config


async def main(job_config_id: str, model_weights_id: str):
    # Load job configuration
    job_config = load_job_config(job_config_id)

    model, dataloader, tokenizer, device = \
        await configure_model_and_dataloader(job_config, for_inference=True, model_weights_id=model_weights_id)

    # Variables to store predictions and actual labels
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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


job_config_id = sys.argv[1]
model_weights_id = sys.argv[2]
asyncio.run(main(job_config_id, model_weights_id))
