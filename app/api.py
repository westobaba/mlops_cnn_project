import mlflow.pytorch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torchvision.transforms as transforms
import torch

app = FastAPI(title="CNN Image Classification API")

# -----------------------------
# Helper to load the latest MLflow model
# -----------------------------
def load_latest_model(experiment_name="Default", artifact_name="cnn_model"):
    # Get experiment
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")

    # Get latest run
    runs = client.list_run_infos(experiment.experiment_id)
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'.")

    latest_run_id = runs[-1].run_id  # newest run is last
    model_uri = f"runs:/{latest_run_id}/{artifact_name}"

    print(f"🔹 Loading model from run_id: {latest_run_id}")
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model

# Load model once at startup
model = load_latest_model()

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
        _, predicted = torch.max(outputs, 1)
    
    return {"class": str(predicted.item())}
