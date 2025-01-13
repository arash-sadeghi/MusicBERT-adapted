# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04

# Install Python 3.7
RUN apt-get update && apt-get install -y \
    python3.7 python3.7-distutils python3-pip python3.7-dev \
    git curl gcc g++ make \
    libffi-dev libssl-dev libsasl2-dev libldap2-dev libpython3.7-dev \
    libopenblas-dev liblapack-dev libsndfile1-dev libjpeg-dev zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.7 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Upgrade pip, setuptools, wheel, and install Cython
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir Cython==0.29.36

# Install numpy separately to satisfy build dependencies
RUN python -m pip install --no-cache-dir numpy==1.21.6

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Create a working directory
WORKDIR /app

# Install Python dependencies
RUN pip3 install torch

# Copy application files
COPY requirements.txt /app

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY musicbert /app/musicbert
COPY *.py /app
COPY checkpoint_last_musicbert_base.pt  /app
COPY dict.json  /app
COPY standalone_musicbert_model.pth /app
COPY torch_groove_val.pkl /app
COPY torch_groove_train.pkl /app
COPY processed.txt /app

# Expose any required ports
EXPOSE 5000

# Run the application
CMD ["python", "trainer.py"]
