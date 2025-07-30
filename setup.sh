#!/bin/bash

# Create necessary directories
mkdir -p ~/.streamlit/

# Create Streamlit config file
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\n\
[browser]\n\
serverAddress = \"0.0.0.0\"\n\
" > ~/.streamlit/config.toml

# Install system dependencies if running on Linux
if [ "$(uname)" == "Linux" ]; then
    apt-get update
    xargs -a packages.txt apt-get install -y
fi

# Ensure pip is up to date
python -m pip install --upgrade pip

# Install Python dependencies
if [ -f "requirements-deploy.txt" ]; then
    echo "Installing minimal deployment requirements..."
    pip install -r requirements-deploy.txt
else
    echo "Error: requirements-deploy.txt not found!"
    exit 1
fi