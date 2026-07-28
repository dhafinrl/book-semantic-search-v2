FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (libomp-dev is often required by FAISS on Linux)
RUN apt-get update && apt-get install -y libomp-dev && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Entrypoint will be defined in docker-compose.yml
