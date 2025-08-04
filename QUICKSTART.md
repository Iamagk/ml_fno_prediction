# 🚀 Quick Start Deployment Guide

## Step 1: Deploy Backend with Docker

```bash
# Navigate to your project directory
cd ml_fno_prediction

# Build and run the backend
docker-compose up --build -d

# Verify it's working
curl http://localhost:8000/
```

Your backend will be available at `http://localhost:8000`

## Step 2: Deploy to Cloud (Choose one)

### Option A: Railway (Recommended)
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Select "Deploy from GitHub repo"
4. Set root directory to `server`
5. Your API will be live at `https://your-app.railway.app`

### Option B: Render
1. Go to [render.com](https://render.com)
2. Create new Web Service
3. Connect GitHub repository
4. Set root directory to `server`
5. Your API will be live at `https://your-app.onrender.com`

## Step 3: Deploy Frontend to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repository
3. Set root directory to `client`
4. Add environment variable:
   - Name: `REACT_APP_API_URL`
   - Value: `https://your-backend-url.com` (from step 2)
5. Deploy

## Step 4: Update CORS (Important!)

Update your backend's CORS settings in `server/index.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend-app.vercel.app",  # Your Vercel URL
        "http://localhost:3000"  # For local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Step 5: Test Your Deployment

- Backend: Visit `https://your-backend-url.com`
- Frontend: Visit `https://your-frontend-app.vercel.app`
- Test API calls from frontend

## 🛠️ Local Development

```bash
# Backend
cd server
pip install -r requirements.txt
uvicorn index:app --reload

# Frontend
cd client
npm install
npm start
```

## 📱 Mobile-Friendly Testing

Your app will be accessible on mobile devices through the Vercel URL. Test on different devices to ensure responsiveness.

## 🔧 Troubleshooting

### CORS Errors
- Update `allow_origins` in your FastAPI app
- Ensure your Vercel URL is included

### API Not Accessible
- Check if your cloud service is running
- Verify the API URL in your frontend environment variables
- Test API endpoints directly with curl or Postman

### Build Failures
- Check that all required files are in the repository
- Verify requirements.txt includes all dependencies
- Ensure model files are committed to the repository

## 📞 Need Help?

1. Check container logs: `docker logs [container-name]`
2. Test API health: `./health-check.sh`
3. Verify environment variables are set correctly
