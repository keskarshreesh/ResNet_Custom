import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

import os

from resnet_core import ResNet50, ResidualBlock
from basic_dataloader import get_train_loader, get_val_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNet50(ResidualBlock, [3, 4, 6, 3], 45)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

num_epochs = 100  # Adjust as per your need
best_val_loss = float('inf')
checkpoint_path = '../model_checkpoints'

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

train_loader = get_train_loader()
val_loader = get_val_loader()

print("Starting training...")

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch}/{num_epochs - 1}, Train Loss: {epoch_loss:.4f}')

    # Validation phase
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

    epoch_val_loss = running_loss / len(val_loader.dataset)
    print(f'Epoch {epoch}/{num_epochs - 1}, Validation Loss: {epoch_val_loss:.4f}')

    # Checkpoint saving
    if epoch_val_loss < best_val_loss:
        print(f"Validation loss decreased from {best_val_loss:.4f} to {epoch_val_loss:.4f}, saving model...")
        best_val_loss = epoch_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,
        }, os.path.join(checkpoint_path, f'model_epoch_{epoch}.pt'))

    # Step the scheduler
    scheduler.step()

