import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torchvision.transforms as transforms
import boto3
import os

from src.model import SimpleCNN  # your CNN model class

app = FastAPI(title="CNN Image Classification API")

# -----------------------------
# Define class labels
# -----------------------------
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]  # Adjust according to your dataset


# -----------------------------
# Download model from S3 if not local
# -----------------------------
def download_from_s3(bucket_name, object_name, file_path):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )
    if not os.path.exists(file_path):
        try:
            s3.download_file(bucket_name, object_name, file_path)
            print(f"☁️ Downloaded {object_name} from s3://{bucket_name} to {file_path}")
        except Exception as e:
            print(f"❌ Could not download model from S3: {e}")


# -----------------------------
# Load model
# -----------------------------
def load_model(bucket_name="your-s3-bucket", object_name="cnn_model.pth"):
    model_path = "cnn_model.pth"

    # Download from S3 if missing
    download_from_s3(bucket_name, object_name, model_path)

    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


# Load model once at startup
model = load_model(bucket_name="your-s3-bucket")


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    
    predicted_class = class_names[predicted_idx.item()]
    return {"class": predicted_class}
