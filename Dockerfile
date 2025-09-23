FROM python:3.10-slim

WORKDIR /app

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU + torchvision
RUN pip install --no-cache-dir \
    torch==2.0.0+cpu torchvision==0.15.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install other packages including python-multipart for FastAPI file uploads
RUN pip install --no-cache-dir fastapi uvicorn pillow tqdm python-multipart "numpy<2"

# Copy project files including cnn_model.pth
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
