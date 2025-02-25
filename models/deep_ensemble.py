import os, sys
sys.path.append("..")
import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
from models.baseline_cnn import BaselineCNN
from data.load_data import load_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train_model(model, train_loader, epochs, device, model_path, weight_init=None):
    """
    Train a single model with optional weight initialization.
    """
    model.to(device)
    if weight_init:
        model.apply(weight_init)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        epoch_loss /= len(train_loader)
        accuracy = correct / total
        scheduler.step(epoch_loss)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), model_path)


def xavier_init(m):
    """ Xavier (Glorot) initialization """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)

def he_init(m):
    """ He initialization (for ReLU-based networks) """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

def normal_init(m):
    """ Normal distribution initialization """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.1)

# Step 4: Train and Save an Ensemble of Models with Data & Initialization Variation
def train_and_save_ensemble(dataset_name, num_models=5, epochs=5, batch_size=128, save_dir="checkpoints/ensemble_models", weight_method="xavier", data_variation=True):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processes = []
    weight_init_methods = {"xavier": xavier_init, "he": he_init, "normal": normal_init}
    selected_weight_init = weight_init_methods.get(weight_method, None)

    for i in range(num_models):
        print(f" Training model {i+1}/{num_models} with {weight_method} initialization...")

        model = BaselineCNN()
        model_path = os.path.join(save_dir, f"model_{i}_{set_seed(10 + i)}.pth")

        train_loader, _ = load_dataset(dataset_name, batch_size) if data_variation else load_dataset(dataset_name, batch_size, fixed=True)

        p = mp.Process(target=train_model, args=(model, train_loader, epochs, device, model_path, selected_weight_init))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All ensemble models trained and saved successfully!")

# Step 5: Load Trained Models for Inference
def load_ensemble_models(num_models=5, save_dir="../checkpoints/ensemble_models", device="cpu"):
    models = []
    for i in range(num_models):
        set_seed
        model = BaselineCNN()
        model.load_state_dict(torch.load(os.path.join(save_dir, f"model_{i}.pth"), map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    return models

# Step 6: Ensemble Prediction with Performance Metrics
def ensemble_predict(models, test_loader, device):
    """
    Perform inference using ensemble models and return:
    - Mean prediction (softmax average)
    - Predictive variance (uncertainty estimation)
    - Accuracy, F1-Score, and Recall
    """
    softmax = nn.Softmax(dim=1)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=">à Making Ensemble Predictions"):
            images = images.to(device)
            batch_preds = torch.stack([softmax(model(images)) for model in models])
            mean_preds = batch_preds.mean(dim=0)
            variance = batch_preds.var(dim=0)

            all_preds.append(mean_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.hstack(all_labels)

    predicted_classes = np.argmax(all_preds, axis=1)

    accuracy = accuracy_score(all_labels, predicted_classes)
    f1 = f1_score(all_labels, predicted_classes, average="weighted")
    recall = recall_score(all_labels, predicted_classes, average="weighted")
    precision = precision_score(all_labels, predicted_classes, average="weighted")

    print(f"Ensemble Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    return accuracy, f1, recall, precision, variance

# Step 7: Visualize Predictive Variance
def plot_variance(variance, title="Predictive Variance Distribution"):
    var_mean = variance.mean(dim=1).cpu().numpy()

    plt.figure(figsize=(8, 5))
    sns.histplot(var_mean, bins=20, kde=True)
    plt.title(title)
    plt.xlabel("Variance (Uncertainty)")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == "__main__":
    datasets = ["mnist"]
    for dataset in datasets:
        train_and_save_ensemble(dataset, num_models=5, epochs=5, batch_size=128, save_dir="../checkpoints/ensemble_models", weight_method="xavier", data_variation=True)
        models = load_ensemble_models(num_models=5, save_dir="../checkpoints/ensemble_models", device="cpu")
        test_loader, _ = load_dataset(dataset, batch_size=128, train=False)
        accuracy, f1, recall, precision, variance = ensemble_predict(models, test_loader, device="cpu")
        # save results
        np.save(f"../results/{dataset}_variance.npy", variance.cpu().numpy())
        plot_variance(variance, title=f"{dataset.upper()} Predictive Variance Distribution")
    