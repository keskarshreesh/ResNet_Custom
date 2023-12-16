import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([
    transforms.Resize((224,224)),  # Resize to 224x224
    transforms.ToTensor(),   # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

train_dataset = datasets.ImageFolder(root='/common/users/skk139/ResNet_Custom/datasets/fruits/fruits-360_dataset/fruits-360/Training', transform=transform)
test_dataset_full = datasets.ImageFolder(root='/common/users/skk139/ResNet_Custom/datasets/fruits/fruits-360_dataset/fruits-360/Test', transform=transform)

batch_size = 16

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

total_test_size = len(test_dataset_full)
test_size = len(total_test_size)*0.6
valid_size = total_test_size - test_size

test_dataset, valid_dataset = random_split(test_dataset_full, [test_size, valid_size])

val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def get_train_loader():
    return train_loader

def get_test_loader():
    return test_loader

def get_val_loader():
    return val_loader