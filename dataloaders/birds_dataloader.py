from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(224),  # Resize to 224x224
    transforms.ToTensor(),   # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

train_dataset = datasets.ImageFolder(root='/common/users/skk139/ResNet_Custom/Aishwarya/ResNet_Custom/datasets/birds/train', transform=transform)
test_dataset = datasets.ImageFolder(root='/common/users/skk139/ResNet_Custom/Aishwarya/ResNet_Custom/datasets/birds/test', transform=transform)
valid_dataset = datasets.ImageFolder(root='/common/users/skk139/ResNet_Custom/Aishwarya/ResNet_Custom/datasets/birds/valid', transform=transform)

batch_size = 16

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def get_train_loader():
    return train_loader

def get_test_loader():
    return test_loader

def get_val_loader():
    return val_loader