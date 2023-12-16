import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os

from resnet_core import ResNet50, ResidualBlock
import dataloaders.basic.fruits_dataloader, dataloaders.augmented.fruits_dataloader

mode_augment = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = len(os.listdir("/common/users/skk139/ResNet_Custom/datasets/fruits/fruits-360_dataset/fruits-360/Training"))

model = ResNet50(ResidualBlock, [3, 4, 6, 3], num_classes=num_classes)
model = model.to(device)

writer = SummaryWriter('../runs/experiment_2')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

num_epochs = 100  # Adjust as per your need
best_val_loss = float('inf')
checkpoint_path = '../checkpoints'

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

train_loader = dataloaders.augmented.fruits_dataloader.get_train_loader() if mode_augment else dataloaders.basic.fruits_dataloader.get_train_loader()
val_loader = dataloaders.augmented.fruits_dataloader.get_val_loader() if mode_augment else dataloaders.basic.fruits_dataloader.get_val_loader()

print("Starting training...")

total_batches = len(train_loader)

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        percent_complete = 100. * (i + 1) / total_batches
        print(f'Epoch {epoch+1}, {percent_complete:.2f}% complete')

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch}/{num_epochs - 1}, Train Loss: {epoch_loss:.4f}')
    writer.add_scalar('Training Loss', epoch_loss, epoch)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch}/{num_epochs - 1}, Train Loss: {epoch_loss:.4f}')
    writer.add_scalar('Training Loss', epoch_loss, epoch)

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
    writer.add_scalar('Validation Loss', epoch_val_loss, epoch)

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