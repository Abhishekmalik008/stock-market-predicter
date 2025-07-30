# Stock Market Predictor - Deployment Guide

This guide will help you deploy the Stock Market Predictor application.

## Option 1: Deploy to Streamlit Cloud (Recommended)

1. Create a GitHub repository for your project if you haven't already.
2. Push your code to the repository.
3. Go to [Streamlit Cloud](https://share.streamlit.io/)
4. Click on "New app" and connect your GitHub repository.
5. Configure the following settings:
   - Repository: Select your repository
   - Branch: main (or your preferred branch)
   - Main file path: `main.py`
   - Python version: 3.10
6. Click "Deploy!"

## Option 2: Deploy to Heroku

1. Install the Heroku CLI from https://devcenter.heroku.com/articles/heroku-cli
2. Run the following commands:
   ```bash
   # Login to Heroku
   heroku login
   
   # Create a new Heroku app
   heroku create your-app-name
   
   # Set Python version
   echo "python-3.10.0" > runtime.txt
   
   # Create a Procfile with the following content:
   # web: sh setup.sh && streamlit run main.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false
   
   # Deploy your application
   git add .
   git commit -m "Prepare for Heroku deployment"
   git push heroku main
   ```

## Option 3: Local Development

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application locally:
   ```bash
   streamlit run main.py
   ```

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# API Keys (if any)
# ALPHA_VANTAGE_API_KEY=your_api_key_here
# FINNHUB_API_KEY=your_api_key_here
```

## Troubleshooting

- If you encounter memory issues during deployment, try adding a `.streamlit/config.toml` file with:
  ```toml
  [server]
  maxUploadSize = 500
  ```

- For Streamlit Cloud deployment issues, check the logs in the Streamlit Cloud dashboard.

## Notes

- The application requires Python 3.10 or higher.
- Some features might require API keys for data providers.
- For production use, consider implementing proper authentication and rate limiting.
