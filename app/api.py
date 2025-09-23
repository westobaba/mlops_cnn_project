import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torchvision.transforms as transforms

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
# Load model
# -----------------------------
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

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
