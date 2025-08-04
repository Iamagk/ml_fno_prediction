#!/bin/bash

# Deployment script for Docker

set -e

echo "Building Docker image..."
docker build -t ml-fno-prediction-api ./server

echo "Stopping existing container if running..."
docker stop ml-fno-api || true
docker rm ml-fno-api || true

echo "Running new container..."
docker run -d \
  --name ml-fno-api \
  -p 8000:8000 \
  --restart unless-stopped \
  ml-fno-prediction-api

echo "Checking container status..."
docker ps | grep ml-fno-api

echo "API should be available at http://localhost:8000"
echo "Health check: curl http://localhost:8000/"
