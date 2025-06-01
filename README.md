# 🧠 Cerebrium Image Classification Deployment

A complete Machine Learning deployment solution for image classification on **Cerebrium's serverless GPU platform**.  
This project converts a **PyTorch** model to **ONNX** format and deploys it using **Docker** containers on Cerebrium, with comprehensive testing and monitoring capabilities.

---

## 🎯 Project Overview

This repository implements a **production-ready image classification pipeline** that:

- ✅ Converts PyTorch models to ONNX format for optimized inference  
- 🚀 Deploys the model on Cerebrium's serverless GPU platform using Docker  
- 🧪 Provides comprehensive testing for both local development and deployed models  
- ⚡ Ensures sub-3-second response times for production requirements  
- 📊 Includes monitoring and error handling for production deployment  

---

## 🏗️ Architecture

The deployment pipeline consists of the following stages:

1. **Model Development (Local)**
   - Train and export a model using **PyTorch**

2. **Model Conversion**
   - Convert the PyTorch model to **ONNX** for optimized inference

3. **Containerization**
   - Package the ONNX model and inference code in a **Docker** container (`Dockerfile`)

4. **Deployment to Cerebrium**
   - Deploy the Docker container to **Cerebrium’s serverless GPU platform**

5. **Testing**
   - Perform local testing using `test.py`
   - Perform post-deployment testing via `test_server`


## 🚀 Quick Start

### 🔧 Prerequisites

Make sure you have the following installed:

- Python 3.8 or higher  
- Docker (for deployment)  
- Cerebrium account with API access  
- Git (for version control)

---

### 1️⃣ Environment Setup

```bash
# Clone the repository
git clone https://github.com/SoberSalman/cerebrium-image-classification-deployment
cd cerebrium-classification-deploy

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt



### 2️⃣ Model Conversion

Convert the PyTorch model to ONNX format:

```bash
# (Optional) Download pre-trained weights if not already present
# wget https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth

# Convert PyTorch model to ONNX
python src/convert_to_onnx.py
```

✅ This will create `image_classification_model.onnx` in your project directory.

---

### 3️⃣ Local Testing

Test the converted model locally:

```bash
# Run comprehensive local tests
python tests/test.py
```

---

### 4️⃣ Deploy to Cerebrium

```bash
# Install Cerebrium CLI (if not already installed)
pip install cerebrium

# Login to Cerebrium
cerebrium login

# Deploy the model
cerebrium deploy
```

---

### 5️⃣ Test Deployment

Test the deployed model on Cerebrium:

```bash
# Test single image classification
python tests/test_server.py assets/n01440764_tench.jpg

# Run comprehensive deployment tests
python tests/test_server.py --preset-tests
```
## 🐳 Cerebrium Deployment Configuration

### Cerebrium Platform Setup

This project is deployed on Cerebrium's serverless GPU platform using the following configuration:

#### `cerebrium.toml` Configuration

```toml
[cerebrium.deployment]
name = "image-classification-deploy"
python_version = "3.9"
include = ["./*"]
exclude = [
    "venv/*",
    ".git/*",
    "*.pyc",
    "__pycache__/*",
    ".pytest_cache/*",
    "*.log",
    "logs/*",
    "models/pytorch_model_weights.pth",  # Downloaded during build
    "models/model.onnx"                   # Generated during build
]

[cerebrium.hardware]
cpu = 2
memory = 4.0
compute = "CPU"                    # CPU-optimized for ONNX inference
gpu_count = 0
provider = "aws"
region = "us-east-1"

[cerebrium.runtime.custom]
port = 8000
healthcheck_endpoint = "/health"
dockerfile_path = "./Dockerfile"

[cerebrium.scaling]
min_replicas = 0                   # Serverless - scales to zero
max_replicas = 5                   # Auto-scales based on demand
cooldown = 30                      # Seconds between scaling decisions
replica_concurrency = 100         # Concurrent requests per replica

```

## Production Docker Configuration

The deployment uses a multi-stage, optimized Docker image:

```dockerfile
# Use Python base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies for image processing and ML
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch CPU-optimized version first
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/pytorch_model.py ./models/
COPY main.py .

# Create required directories
RUN mkdir -p /app/models /app/logs /app/tmp /app/assets

# Download pre-trained model weights during build
RUN wget -O /app/models/pytorch_model_weights.pth "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1"

# Convert PyTorch model to ONNX during build (optimization step)
RUN python /app/src/convert_to_onnx.py

# Set proper permissions
RUN chmod -R 755 /app

# Expose application port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
```

## Detailed Usage Examples

### 1. Single Image Classification

```bash
# Test with tench image (should return class 0)
python tests/test_server.py assets/n01440764_tench.jpeg
# Output: 0

# Test with turtle image (should return class 35)
python tests/test_server.py assets/n01667114_mud_turtle.JPEG
# Output: 35

# Test with your own image
python tests/test_server.py path/to/your/image.jpg
# Output: <predicted_class_id>

## 2. Comprehensive Testing

```bash
# Run all preset tests
python tests/test_server.py --preset-tests
```