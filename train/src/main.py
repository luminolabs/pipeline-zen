import asyncio
import os
import sys
from datetime import datetime

import torch
from torch import nn, optim, Tensor

from common.loss.utils import loss_factory
from common.utils import get_model_weights_path, load_job_config
from common.helpers import configure_model_and_dataloader
from common.tokenizer.utils import tokenize_inputs


async def main(job_config_id: str):
    # Load job configuration
    job_config = load_job_config(job_config_id)

    model, dataloader, tokenizer, device = \
        await configure_model_and_dataloader(job_config)

    # TODO: Implement metrics lib, to capture timing, model, etc metrics
    # TODO: Use logger instead of print

    # Loss calculator
    criterion = loss_factory(
        job_config.get('loss_func_name'),
        **job_config.get('loss_func_args', {}))
    # criterion = CrossEntropyLoss()
    # Optimizer
    # TODO: Allow using different optimizers through configuration
    optimizer = optim.Adam(model.parameters(), lr=job_config.get('learning_rate'))
    print("Loss and Optimizer is set")

    # Capture the start time
    start_time = datetime.now()
    print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for epoch_cnt in range(job_config.get('num_epochs')):
        running_loss = 0.0
        batch_cnt = 0
        model.train()
        # The dataloader will load a batch of records from the dataset
        for inputs, labels in dataloader:
            batch_cnt += 1
            model_args = {}
            if tokenizer:
                inputs = tokenize_inputs(inputs, tokenizer, model_args, device)
            # Load inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Run training logic
            optimizer.zero_grad()
            outputs = model(inputs, **model_args)
            if isinstance(outputs, Tensor):
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Log training information
            print(f'Batch {batch_cnt}/{len(dataloader)}, Batch Loss: {loss.item()}')
            # Exit if `num_batches` is reached. This option is used when testing,
            # to stop training loop before the actual end of the dataset is reached
            if job_config.get('num_batches') == batch_cnt:
                break
        # Log training information
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
    model_weights_path = get_model_weights_path(job_config.get('job_id'))
    torch.save(model.module.state_dict()
               if isinstance(model, nn.DataParallel)
               else model.state_dict(), model_weights_path)
    print("Trained model saved! at: " + model_weights_path)
    print("... use these arguments to evaluate your model: `" +
          job_config_id + " " +
          os.path.basename(model_weights_path) + "`")


job_config_id = sys.argv[1]
asyncio.run(main(job_config_id))
