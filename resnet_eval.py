import torch

from resnet_core import ResNet50, ResidualBlock
from dataloaders.resisc_dataloader import get_test_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNet50(ResidualBlock, [3, 4, 6, 3], 45)
model = model.to(device)

model.eval()

correct = 0
total = 0

test_loader = get_test_loader()

# No gradient is needed for evaluation
with torch.no_grad():
    for images, labels in test_loader:
        # Forward pass to get outputs
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
