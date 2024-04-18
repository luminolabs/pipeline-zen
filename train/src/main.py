import asyncio
import os
from datetime import datetime

import torch
from torch import nn, optim
from transformers import TensorType
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy

from common.utils import configure_model_and_dataloader
from imdb_sentiment import job_config_


async def main():
    # Load job configuration
    job_config = job_config_

    model, dataloader, tokenizer, device = \
        await configure_model_and_dataloader(job_config)

    # TODO: Implement metrics lib, to capture timing, model, etc metrics
    # TODO: Use logger instead of print

    # Loss calculator
    # TODO: Allow using different loss calculators through configuration
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    # TODO: Allow using different optimizers through configuration
    optimizer = optim.Adam(model.parameters(), lr=job_config.get('learning_rate'))
    print("Loss and Optimizer is set")

    # Capture the start time
    start_time = datetime.now()
    print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch_cnt in range(job_config.get('num_epochs')):
        running_loss = 0.0
        batch_cnt = -1
        model.train()
        # The dataloader will load a batch of records from the dataset
        for inputs, labels in dataloader:
            batch_cnt += 1
            # Attention masks are only used with tokenized inputs
            attention_masks = None
            if tokenizer:
                # Tokenize batch of inputs
                # Tensor data need to be of same length, so we need to
                # set max size and padding options
                tokenized_values = tokenizer(
                    inputs,
                    padding=PaddingStrategy.MAX_LENGTH,
                    truncation=TruncationStrategy.ONLY_FIRST,
                    max_length=tokenizer.model_max_length,
                    return_tensors=TensorType.PYTORCH
                )
                # Replace original inputs with tokenized inputs
                inputs = tokenized_values.get('input_ids')
                # Load attention masks to device
                attention_masks = tokenized_values.get('attention_mask').to(device)

            # Load inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Run training logic
            optimizer.zero_grad()
            outputs = model(inputs, attention_mask=attention_masks)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Log training information
            print(f'Batch {batch_cnt + 1}/{len(dataloader)}, Batch Loss: {loss.item()}')
            # Exit if `num_batches` is reached. This option is used when testing,
            # to stop training loop before the actual end of the dataset is reached
            if job_config.get('num_batches'):
                break
    print(f'Epoch {epoch_cnt + 1}/{job_config.get("num_epochs")}, Loss: {running_loss / len(dataloader)}')

    # Capture the end time
    end_time = datetime.now()
    print(f"Training ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate and print the total training time
    total_time = end_time - start_time
    total_minutes = total_time.total_seconds() / 60
    print(f"Total training time: {total_minutes:.2f} minutes")

    # TODO: Implement different storage strategies; ex. gcp/s3 bucket
    print("Training loop complete, now saving the model")
    # Save the trained model
    model_path = job_config.get('model_weights_path')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print("Trained model saved!")


asyncio.run(main())
