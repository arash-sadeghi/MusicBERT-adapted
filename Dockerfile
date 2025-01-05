# Use an official Python image as base
FROM python:3.7-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc g++ make libffi-dev libssl-dev libsasl2-dev libldap2-dev python3-dev cython3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*



# Create a working directory
WORKDIR /app

# Copy application files
COPY musicbert /app/musicbert
COPY *.py /app
COPY checkpoint_last_musicbert_base.pt  /app
COPY dict.json  /app
COPY requirements.txt /app
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY standalone_musicbert_model.pth /app
COPY torch_groove_val.pkl /app
COPY torch_groove_train.pkl /app
# Expose any required ports (optional, e.g., Flask app on port 5000)
EXPOSE 5000

# Run the application
CMD ["python", "trainer.py"]

