#!/bin/bash

echo "========================================"
echo "AI Waste Segregation - Backend Setup"
echo "========================================"
echo

cd backend

echo "Creating virtual environment..."
python3 -m venv venv

echo
echo "Activating virtual environment..."
source venv/bin/activate

echo
echo "Installing dependencies..."
pip install -r requirements.txt

echo
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo
echo "To start the backend server:"
echo "  1. cd backend"
echo "  2. source venv/bin/activate"
echo "  3. uvicorn app.main:app --reload"
echo
