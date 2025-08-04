// Configuration for different environments
const config = {
  production: {
    API_URL: "https://ml-fno-prediction-server.onrender.com"
  },
  development: {
    API_URL: "http://localhost:8000"
  }
};

// Automatically detect environment
const environment = process.env.NODE_ENV === 'development' ? 'development' : 'production';

export default config[environment];
