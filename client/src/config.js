// Configuration for different environments
const config = {
  production: {
    API_URL: "https://ml-fno-prediction-server.onrender.com"
  },
  development: {
    API_URL: "https://ml-fno-prediction-server.onrender.com"  // Use production for now until ensemble pricing is fixed locally
  }
};

// Automatically detect environment
const environment = process.env.NODE_ENV === 'development' ? 'development' : 'production';

export default config[environment];
