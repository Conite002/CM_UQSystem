import torch
import torchvision
import torchvision.transforms as transforms

def load_mnist(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_cifar10(batch_size=128):
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
    
def load_dataset(dataset, batch_size=128, train=True):
    if dataset == "mnist":
        return load_mnist(batch_size)
    elif dataset == "cifar10":
        return load_cifar10(batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")