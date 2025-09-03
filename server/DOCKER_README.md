# Docker Setup for ML FNO Prediction Server

This directory contains everything needed to run your ML FNO Prediction Server in a Docker container.

## Quick Start

### Option 1: Using the build script (Recommended)
```bash
cd server
./build_and_run.sh
```

This script will:
- Build the Docker image
- Stop any existing container
- Start a new container
- Test the server

### Option 2: Using Docker Compose
```bash
cd server
docker-compose up -d
```

### Option 3: Manual Docker commands
```bash
cd server

# Build the image
docker build -t ml-fno-server .

# Run the container
docker run -d \
    --name ml-fno-server \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models:ro \
    -v $(pwd)/data:/app/data:ro \
    --restart unless-stopped \
    ml-fno-server
```

## Container Management

### View logs
```bash
docker logs ml-fno-server
```

### Stop the server
```bash
docker stop ml-fno-server
```

### Restart the server
```bash
docker restart ml-fno-server
```

### Remove the container
```bash
docker rm ml-fno-server
```

### Remove the image
```bash
docker rmi ml-fno-server
```

## Accessing the Server

Once running, your server will be available at:
- **Local**: http://localhost:8000
- **API Endpoint**: http://localhost:8000/
- **Health Check**: The container includes a health check that monitors the root endpoint

## API Endpoints

- `GET /` - Home/health check
- `GET /fetch_yfinance?symbol={symbol}` - Fetch stock data
- `GET /predict_live?symbol={symbol}` - Get live prediction
- `POST /predict` - Submit prediction request
- `GET /predict_with_options?symbol={symbol}` - Get prediction with options pricing
- `GET /debug_expiry` - Debug expiry date calculation

## Volumes

The container mounts:
- `./models` - Read-only access to your trained models
- `./data` - Read-only access to data files

## Environment Variables

- `PYTHONUNBUFFERED=1` - Ensures Python output is not buffered
- `PORT=8000` - Port the server runs on

## Troubleshooting

### Port already in use
If port 8000 is already in use, you can change it in the docker-compose.yml or docker run command:
```bash
docker run -d --name ml-fno-server -p 8001:8000 ml-fno-server
```

### Permission issues
Make sure the models and data directories are readable by the Docker process.

### Model file not found
Ensure your trained model files are in the `./models` directory and accessible.

## Building for Production

For production deployment, consider:
1. Using a multi-stage build to reduce image size
2. Adding environment-specific configuration
3. Setting up proper logging and monitoring
4. Using Docker secrets for sensitive data
