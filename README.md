# MLOps CNN Project
# 🖼️ CNN Image Classification API (MLOps Deployment)

This project demonstrates how to **train, package, and deploy** a **Convolutional Neural Network (CNN)** for image classification using:

- **PyTorch** (Model training & inference)
- **FastAPI** (REST API service)
- **Docker** (Containerization)
- **AWS ECR** (Image registry)
- **AWS S3** (Model storage)
- **AWS EC2** (Deployment)

The deployed API accepts an image and returns the predicted class (CIFAR-10 dataset: airplane, dog, cat, etc.).

---

## 📖 Table of Contents
1. [Features](#-features)
2. [Architecture](#-architecture)
3. [Project Structure](#-project-structure)
4. [Setup & Installation](#-setup--installation)
   - Local Development
   - Docker
   - AWS ECR & EC2
5. [Running the API](#-running-the-api)
6. [Testing the API](#-testing-the-api)
7. [Deployment Pipeline](#-deployment-pipeline)
8. [Troubleshooting](#-troubleshooting)
9. [Future Improvements](#-future-improvements)
10. [License](#-license)

---

## 🚀 Features
- **CNN Model** trained on CIFAR-10
- **REST API** for predictions
- **Dockerized** for reproducibility
- **MLOps Ready** → model pulled from AWS S3, container deployed via AWS ECR & EC2
- **Scalable** → can be extended with CI/CD pipelines

---

## 🏗️ Architecture
```text
[ Training (local/Colab) ] 
        │
        ▼
[ Save model → S3 bucket ]
        │
        ▼
[ Build Docker image with FastAPI + Model loader ]
        │
        ▼
[ Push to AWS ECR ]
        │
        ▼
[ Deploy on EC2 instance via Docker ]
        │
        ▼
[ Public FastAPI Endpoint → /predict/ ]

mlops_cnn_project/
│── app/
│   ├── api.py          # FastAPI app
│── src/
│   ├── model.py        # Simple CNN architecture
│── Dockerfile          # Container build instructions
│── requirements.txt    # Python dependencies
│── cnn_model.pth       # Trained model (backup, main copy in S3)
│── README.md           # Project documentation


# Clone repo
git clone https://github.com/<your-repo>/mlops_cnn_project.git
cd mlops_cnn_project

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run FastAPI locally
uvicorn app.api:app --host 0.0.0.0 --port 8000
