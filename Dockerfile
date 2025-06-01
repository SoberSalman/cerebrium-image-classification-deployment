# Use Python base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
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

# Install PyTorch CPU first
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements (without torch/torchvision)
COPY requirements.txt .

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/pytorch_model.py ./models/
COPY main.py .

# Create directories
RUN mkdir -p /app/models /app/logs /app/tmp /app/assets

# Download model weights
RUN wget -O /app/models/pytorch_model_weights.pth "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1"

# Convert PyTorch model to ONNX
RUN python /app/src/convert_to_onnx.py

# Set permissions
RUN chmod -R 755 /app

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
