#!/bin/bash
set -e

echo "ChurnSense Startup"
echo "=================="

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

echo "Starting API server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
