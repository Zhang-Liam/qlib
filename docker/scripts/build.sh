#!/bin/bash

# Qlib Production Trading - Docker Build Script

set -e

echo "🐳 Building Qlib Production Trading Docker Image..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the image
echo "📦 Building image..."
docker build -t qlib-prod:latest .

echo "✅ Build completed successfully!"
echo ""
echo "To run the container:"
echo "  docker run -d --name qlib-trading --env-file .env qlib-prod:latest"
echo ""
echo "Or use docker-compose:"
echo "  docker-compose up -d" 