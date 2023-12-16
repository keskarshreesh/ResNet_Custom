import torch
import os
import os
from resnet_core import ResNet50, ResidualBlock
import dataloaders.basic.birds_dataloader, dataloaders.augmented.birds_dataloader

mode_augment = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = len(os.listdir("/common/users/skk139/ResNet_Custom/datasets/birds/train"))

model = ResNet50(ResidualBlock, [3, 4, 6, 3], num_classes=num_classes)
model = model.to(device)

model_checkpoint = torch.load("../checkpoints/model_epoch_95.pt") # Path to be changed

if 'model_state_dict' in model_checkpoint:
    # Load the state dictionary into the model
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model.load_state_dict(model_checkpoint['model_state_dict'])
else:
    # If the checkpoint contains the model directly
    model.load_state_dict(model_checkpoint)

model.eval()

correct = 0
total = 0

test_loader = dataloaders.augmented.resisc_dataloader.get_test_loader() if mode_augment else dataloaders.basic.resisc_dataloader.get_test_loader()

# No gradient is needed for evaluation
with torch.no_grad():
    for images, labels in test_loader:
        # Forward pass to get outputs
        images, labels = images.to(device), labels.to(device)
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        correct += (predicted == labels).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print(f'Accuracy of the model on the test images: {accuracy}%')
