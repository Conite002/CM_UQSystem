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

# if __name__ == "__main__":
#     train_loader, test_loader = load_mnist()
#     print(f"Train batch: {next(iter(train_loader))[0].shape}")
