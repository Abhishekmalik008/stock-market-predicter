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
if [ -f "requirements-streamlit.txt" ]; then
    echo "Installing Streamlit-optimized requirements..."
    pip install -r requirements-streamlit.txt
elif [ -f "requirements-minimal.txt" ]; then
    echo "Installing minimal requirements..."
    pip install -r requirements-minimal.txt
elif [ -f "Pipfile" ]; then
    pip install pipenv
    pipenv install --system --deploy
elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi