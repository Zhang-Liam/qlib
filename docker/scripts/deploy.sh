#!/bin/bash

# Qlib Production Trading - Docker Deployment Script

set -e

echo "🚀 Deploying Qlib Production Trading System..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please copy env.example to .env and configure your settings."
    echo "   cp env.example .env"
    echo "   # Then edit .env with your actual values"
    exit 1
fi

# Build the image
echo "📦 Building Docker image..."
docker build -t qlib-prod:latest .

# Stop and remove existing container if it exists
if docker ps -a --format 'table {{.Names}}' | grep -q "qlib-trading"; then
    echo "🔄 Stopping existing container..."
    docker stop qlib-trading || true
    docker rm qlib-trading || true
fi

# Create necessary directories
mkdir -p logs data config

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

# Wait a moment for container to start
sleep 5

# Check if container is running
if docker ps --format 'table {{.Names}}' | grep -q "qlib-trading"; then
    echo "✅ Deployment completed successfully!"
    echo ""
    echo "📊 Container Status:"
    docker ps --filter "name=qlib-trading"
    echo ""
    echo "📋 Useful Commands:"
    echo "  View logs:     docker logs -f qlib-trading"
    echo "  Stop:          docker stop qlib-trading"
    echo "  Restart:       docker restart qlib-trading"
    echo "  Shell access:  docker exec -it qlib-trading bash"
    echo ""
    echo "🔍 Health Check:"
    docker exec qlib-trading python -c "import qlib.production; print('✅ System is healthy')" || echo "❌ Health check failed"
else
    echo "❌ Deployment failed. Container is not running."
    echo "Check logs with: docker logs qlib-trading"
    exit 1
fi 