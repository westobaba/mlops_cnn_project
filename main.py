import argparse
import sys
import subprocess
from src.train import train_model

def run_training():
    print("🚀 Starting training...")
    train_model(epochs=5, lr=0.001, batch_size=64)

def run_server():
    print("🚀 Starting FastAPI server on http://127.0.0.1:8000")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "app.api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps CNN Project")
    parser.add_argument("mode", choices=["train", "serve"], help="train or serve API")
    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    elif args.mode == "serve":
        run_server()
