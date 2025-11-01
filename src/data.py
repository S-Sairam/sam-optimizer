import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_cifar10_loaders(batch_size):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        ])
    
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    train_dset = datasets.CIFAR10(root='.', train=True, download=True, transform=train_transform)
    test_dset = datasets.CIFAR10(root='.', train=False, download=True, transform=valid_transform)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, test_loader

