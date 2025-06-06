import torchvision.transforms as transforms
from torchvision.datasets import MNIST, SVHN
from torch.utils.data import DataLoader

def get_mnist(batch_size=128, train=True):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # SVHN имеет размер 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = MNIST(root='./data', train=train, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)

def get_svhn(batch_size=128, split='test'):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(),  # приводим к 1-канальному виду
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = SVHN(root='./data', split=split, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
