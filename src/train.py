import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from src.data import get_dataloaders
from src.model import SimpleCNN


def train_model(epochs=5, lr=0.001, batch_size=64):
    device = "cpu"  # <-- Force CPU training

    trainloader, testloader = get_dataloaders(batch_size)
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mlflow.set_experiment("cnn_cifar10_experiment")

    with mlflow.start_run():
        mlflow.log_params({"epochs": epochs, "lr": lr, "batch_size": batch_size})

        for epoch in range(epochs):
            running_loss = 0.0
            progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = running_loss / len(trainloader)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # Evaluation
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        mlflow.log_metric("test_accuracy", accuracy)

        # Save model to MLflow
        mlflow.pytorch.log_model(model, "cnn_model")

    return model
