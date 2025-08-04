#!/bin/bash

# Health check script for the FastAPI backend

API_URL="http://localhost:8000"

echo "🔍 Checking API health..."

# Check if the API is responding
if curl -f -s "$API_URL/" > /dev/null; then
    echo "✅ API is healthy and responding"
    
    # Test a sample prediction endpoint
    echo "🧪 Testing prediction endpoint..."
    if curl -f -s "$API_URL/predict_live?symbol=^NSEI" > /dev/null; then
        echo "✅ Prediction endpoint is working"
    else
        echo "❌ Prediction endpoint is not responding"
    fi
    
else
    echo "❌ API is not responding"
    echo "   Check if the container is running: docker ps | grep ml-fno-api"
    echo "   Check container logs: docker logs ml-fno-api"
fi

echo ""
echo "📊 Container status:"
docker ps | grep ml-fno-api || echo "No container found with name 'ml-fno-api'"

echo ""
echo "📝 Recent logs:"
docker logs --tail 10 ml-fno-api 2>/dev/null || echo "Cannot access container logs"
