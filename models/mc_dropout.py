import torch
import torch.nn as nn
import torch.nn.functional as F

class MCDropoutCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(MCDropoutCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))  # Dropout applied at inference time
        x = self.fc2(x)
        return x

    def enable_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Enable dropout during inference

if __name__ == "__main__":
    model = MCDropoutCNN()
    model.enable_dropout()
    x = torch.randn(1, 1, 28, 28)
    print(model(x).shape)
