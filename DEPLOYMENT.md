# ML FNO Prediction - Deployment Guide

This project consists of a FastAPI backend for ML-based F&O prediction and a React frontend.

## 🚀 Quick Deployment

### Backend (FastAPI with Docker)

#### Option 1: Using Docker Compose (Recommended)
```bash
# Clone the repository
git clone <your-repo-url>
cd ml_fno_prediction

# Build and run with docker-compose
docker-compose up --build -d

# Check if it's running
curl http://localhost:8000/
```

#### Option 2: Using Docker directly
```bash
# Build the Docker image
docker build -t ml-fno-prediction-api ./server

# Run the container
docker run -d -p 8000:8000 --name ml-fno-api ml-fno-prediction-api

# Check if it's running
curl http://localhost:8000/
```

#### Option 3: Using the deployment script
```bash
# Make the script executable and run it
chmod +x deploy.sh
./deploy.sh
```

### Frontend (React on Vercel)

#### Deploy to Vercel
1. Push your code to GitHub
2. Connect your GitHub repository to Vercel
3. Set the environment variable:
   - `REACT_APP_API_URL=https://your-backend-url.com`
4. Deploy

#### Local Development
```bash
cd client
npm install
npm start
```

## 🌐 Cloud Deployment Options for Backend

### Railway
1. Connect your GitHub repository
2. Select the `server` folder as the root directory
3. Railway will automatically detect the Dockerfile
4. Your API will be available at `https://your-app-name.railway.app`

### Render
1. Connect your GitHub repository
2. Create a new Web Service
3. Set root directory to `server`
4. Render will build using the Dockerfile
5. Your API will be available at `https://your-app-name.onrender.com`

### Google Cloud Run
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ml-fno-api ./server

# Deploy to Cloud Run
gcloud run deploy ml-fno-api \
    --image gcr.io/YOUR_PROJECT_ID/ml-fno-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

### AWS ECS with Fargate
1. Push image to ECR
2. Create an ECS service
3. Configure load balancer

## 📁 Project Structure

```
ml_fno_prediction/
├── server/                 # FastAPI backend
│   ├── Dockerfile         # Docker configuration
│   ├── requirements.txt   # Python dependencies
│   ├── index.py          # Main FastAPI application
│   ├── models/           # ML models
│   └── data/             # Data files
├── client/               # React frontend
│   ├── src/
│   │   ├── api.js        # API calls to backend
│   │   └── ...
│   ├── package.json
│   └── .env.example      # Environment variables template
├── docker-compose.yml    # Docker compose configuration
└── deploy.sh            # Deployment script
```

## 🔧 Configuration

### Backend Environment Variables
- `ENV`: Set to "production" for production deployment
- `PORT`: Port number (default: 8000)

### Frontend Environment Variables
- `REACT_APP_API_URL`: Backend API URL

### CORS Configuration
The backend is configured to accept requests from:
- `http://localhost:3000` (local development)
- `https://*.vercel.app` (Vercel deployments)
- All origins (for production - adjust as needed)

## 🚨 Important Notes

1. **Model Files**: Ensure your model files are present in the `server/models/` directory
2. **Data Files**: Make sure required data files are in the `server/data/` directory
3. **API Keys**: Update any API keys (like Calendarific) in your environment
4. **Security**: In production, restrict CORS origins to your specific domains

## 🔍 Health Checks

- Backend health check: `GET /`
- Docker health check included in Dockerfile
- Container will restart automatically if health check fails

## 📝 API Endpoints

- `GET /` - Health check
- `GET /predict_live?symbol={symbol}` - Live prediction
- `GET /predict_with_options?symbol={symbol}` - Prediction with options
- `GET /fetch_yfinance?symbol={symbol}` - Fetch Yahoo Finance data
- `POST /predict` - Prediction with custom features

## 🐛 Troubleshooting

### Backend Issues
- Check container logs: `docker logs ml-fno-api`
- Verify model files exist: `docker exec ml-fno-api ls -la /app/models/`
- Test API directly: `curl http://localhost:8000/`

### Frontend Issues
- Verify environment variables are set correctly
- Check browser network tab for CORS errors
- Ensure API URL is accessible from frontend

### CORS Issues
- Update the `allow_origins` list in `server/index.py`
- Ensure your frontend domain is included

## 📞 Support

If you encounter issues:
1. Check the logs
2. Verify all dependencies are installed
3. Ensure model and data files are present
4. Check network connectivity between frontend and backend
