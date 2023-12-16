import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224,224)),  # Resize to 224x224
    transforms.ToTensor(),   # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

dataset = datasets.ImageFolder(root='/common/users/skk139/ResNet_Custom/datasets/NWPU-RESISC45', transform=transform)

class_indices = [np.where(np.array(dataset.targets) == i)[0] for i in range(len(dataset.classes))]

train_indices = []
val_indices = []
test_indices = []

for indices in class_indices:
    np.random.shuffle(indices)
    train_split = int(0.7 * len(indices))
    val_split = int(0.15 * len(indices))
    test_split = len(indices) - train_split - val_split
    train_indices.extend(indices[:train_split])
    val_indices.extend(indices[train_split:train_split+val_split])
    test_indices.extend(indices[train_split+val_split:train_split+val_split+test_split])

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

batch_size = 16

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def get_train_loader():
    return train_loader

def get_test_loader():
    return test_loader

def get_val_loader():
    return val_loader