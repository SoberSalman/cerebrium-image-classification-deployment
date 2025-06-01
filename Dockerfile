# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY assets/ ./assets/

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp

# Set proper permissions
RUN chmod -R 755 /app

# Download model weights if not present (fallback)
RUN if [ ! -f /app/models/pytorch_model_weights.pth ]; then \
    echo "Model weights not found, please ensure they are copied during build"; \
    fi

# Convert PyTorch model to ONNX (if weights are present)
RUN if [ -f /app/models/pytorch_model_weights.pth ]; then \
    python /app/src/convert_to_onnx.py; \
    fi

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]