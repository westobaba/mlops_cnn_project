import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import boto3
import os

from src.data import get_dataloaders
from src.model import SimpleCNN


def upload_to_s3(file_path, bucket_name, object_name=None):
    """Upload a file to an S3 bucket"""
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )

    if object_name is None:
        object_name = os.path.basename(file_path)

    try:
        s3.upload_file(file_path, bucket_name, object_name)
        print(f"☁️ Uploaded {file_path} to s3://{bucket_name}/{object_name}")
    except Exception as e:
        print(f"❌ S3 Upload failed: {e}")


def train_model(epochs=5, lr=0.001, batch_size=64, bucket_name="cnnmlops"):
    """Train the model and upload to S3"""
    device = "cpu"  # Force CPU training
    trainloader, testloader = get_dataloaders(batch_size)
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
    print(f"✅ Test Accuracy: {accuracy:.4f}")

    # Save model locally
    model_path = "cnn_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved locally as {model_path}")

    # Upload to S3
    upload_to_s3(model_path, bucket_name)

    return model


if __name__ == "__main__":
    # Make sure AWS credentials are set in your environment:
    # export AWS_ACCESS_KEY_ID=xxxx
    # export AWS_SECRET_ACCESS_KEY=xxxx
    # export AWS_DEFAULT_REGION=us-east-1
    train_model(bucket_name="cnnmlops")
