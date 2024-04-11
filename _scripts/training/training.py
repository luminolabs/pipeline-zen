# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import AlzheimerMRIDataset
from model import get_model
from datetime import datetime  # Import datetime to work with times

def train_model():
    # Parameters
    num_classes = 4  # Update this based on your dataset specifics
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Dataset and DataLoader
    print("Loading dataset!")
    train_dataset = AlzheimerMRIDataset(split='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Loading dataset complete!!!")

    # Model
    print("Fetching the model")
    model = get_model(num_classes)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Training on (CPU/GPU?) device:", device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Loss and Optimizer is set")

    # Capture the start time
    start_time = datetime.now()
    print(f"Training started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

    # Capture the end time
    end_time = datetime.now()
    print(f"Training ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate and print the total training time
    total_time = end_time - start_time
    total_minutes = total_time.total_seconds() / 60
    print(f"Total training time: {total_minutes:.2f} minutes")

    print("Training loop complete, now saving the model")
    # Save the trained model
    model_path = 'trained_model.pth'
    torch.save(model.state_dict(), model_path)
    print("Trained model saved!")

if __name__ == "__main__":
    train_model()
