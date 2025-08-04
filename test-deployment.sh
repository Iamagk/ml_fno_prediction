#!/bin/bash

# Complete local deployment test script

set -e

echo "🚀 Starting complete deployment test..."

# Build and run the backend
echo "📦 Building Docker image..."
docker build -t ml-fno-prediction-api ./server

echo "🏃 Starting container..."
docker stop ml-fno-api 2>/dev/null || true
docker rm ml-fno-api 2>/dev/null || true

docker run -d \
  --name ml-fno-api \
  -p 8000:8000 \
  ml-fno-prediction-api

echo "⏳ Waiting for API to start..."
sleep 10

# Test the API
echo "🧪 Testing API endpoints..."

# Test health check
if curl -f -s "http://localhost:8000/" > /dev/null; then
    echo "✅ Health check: PASSED"
else
    echo "❌ Health check: FAILED"
    exit 1
fi

# Test prediction endpoint
echo "🔮 Testing prediction endpoint..."
if curl -f -s "http://localhost:8000/predict_live?symbol=^NSEI" > /dev/null; then
    echo "✅ Prediction endpoint: PASSED"
else
    echo "❌ Prediction endpoint: FAILED"
fi

# Test Yahoo Finance endpoint
echo "📈 Testing Yahoo Finance endpoint..."
if curl -f -s "http://localhost:8000/fetch_yfinance?symbol=^NSEI" > /dev/null; then
    echo "✅ Yahoo Finance endpoint: PASSED"
else
    echo "❌ Yahoo Finance endpoint: FAILED"
fi

echo ""
echo "🎉 Deployment test completed!"
echo "📊 Your API is running at: http://localhost:8000"
echo "📚 API documentation: http://localhost:8000/docs"
echo ""
echo "🌐 Next steps:"
echo "1. Deploy this to a cloud service (Railway, Render, etc.)"
echo "2. Update your frontend's REACT_APP_API_URL"
echo "3. Deploy your frontend to Vercel"
echo ""
echo "🛑 To stop the container: docker stop ml-fno-api"
