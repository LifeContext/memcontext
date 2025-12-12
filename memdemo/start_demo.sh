#!/bin/bash

echo "========================================"
echo "🧠 Memcontext Demo Launcher"
echo "========================================"

# Set working directory
cd /root/autodl-tmp

# Set Python path
export PYTHONPATH=/root/autodl-tmp:$PYTHONPATH

echo "📁 Working directory: $(pwd)"
echo "🐍 Python path: $PYTHONPATH"
echo "========================================"

# Install dependencies if needed
# echo "📦 Installing dependencies..."
cd memcontext-playground/memdemo
# pip install -q -r requirements.txt

echo "🚀 Starting Memcontext Demo..."
echo "🌐 Access the demo at: http://localhost:5000"
echo "🌐 Or access via: http://[your-server-ip]:5000"
echo "========================================"

# Run the application
python app.py 