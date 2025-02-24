import os
import torch
import torch.optim as optim
from models.baseline_cnn import BaselineCNN
from data.load_data import load_mnist
import torch.nn as nn

def train_and_save_ensemble(num_models=5, epochs=5, batch_size=128, save_dir="ensemble_models"):
    os.makedirs(save_dir, exist_ok=True)
    train_loader, _ = load_mnist(batch_size)

    for i in range(num_models):
        print(f"Training model {i+1}/{num_models}...")
        model = BaselineCNN()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{i}.pth"))

if __name__ == "__main__":
    train_and_save_ensemble(num_models=3)
