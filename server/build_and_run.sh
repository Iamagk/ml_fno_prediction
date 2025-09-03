#!/bin/bash

echo "Building and running ML FNO Prediction Server..."

# Build the Docker image
echo "Building Docker image..."
docker build -t ml-fno-server .

if [ $? -eq 0 ]; then
    echo "Docker image built successfully!"
    
    # Stop any existing container
    echo "Stopping any existing container..."
    docker stop ml-fno-server 2>/dev/null || true
    docker rm ml-fno-server 2>/dev/null || true
    
    # Run the container
    echo "Starting the server container..."
    docker run -d \
        --name ml-fno-server \
        -p 8000:8000 \
        -v $(pwd)/models:/app/models:ro \
        -v $(pwd)/data:/app/data:ro \
        --restart unless-stopped \
        ml-fno-server
    
    if [ $? -eq 0 ]; then
        echo "Server started successfully!"
        echo "Server is running on http://localhost:8000"
        echo ""
        echo "Useful commands:"
        echo "  View logs: docker logs ml-fno-server"
        echo "  Stop server: docker stop ml-fno-server"
        echo "  Restart server: docker restart ml-fno-server"
        echo "  Remove container: docker rm ml-fno-server"
        echo ""
        echo "Testing the server..."
        sleep 5
        curl -f http://localhost:8000/ && echo " - Server is responding!" || echo " - Server might still be starting up..."
    else
        echo "Failed to start the container!"
        exit 1
    fi
else
    echo "Failed to build Docker image!"
    exit 1
fi
