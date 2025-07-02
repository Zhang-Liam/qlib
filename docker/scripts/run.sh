#!/bin/bash

# Qlib Production Trading - Docker Run Script

set -e

echo "🚀 Starting Qlib Production Trading..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please copy env.example to .env and configure your settings."
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Stop and remove existing container if it exists
if docker ps -a --format 'table {{.Names}}' | grep -q "qlib-trading"; then
    echo "🔄 Stopping existing container..."
    docker stop qlib-trading || true
    docker rm qlib-trading || true
fi

# Create logs and data directories if they don't exist
mkdir -p logs data

# Run the container
echo "🐳 Starting container..."
docker run -d \
    --name qlib-trading \
    --restart unless-stopped \
    --env-file .env \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/config:/app/config \
    qlib-prod:latest

echo "✅ Container started successfully!"
echo ""
echo "To view logs:"
echo "  docker logs -f qlib-trading"
echo ""
echo "To stop the container:"
echo "  docker stop qlib-trading"
echo ""
echo "To restart:"
echo "  docker restart qlib-trading" 