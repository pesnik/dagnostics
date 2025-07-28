#!/bin/bash

set -e

echo "🚀 Deploying DAGnostics..."

# Build and start services
docker-compose build
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Pull Ollama model
echo "Pulling Ollama model..."
docker-compose exec ollama ollama pull mistral

# Check service health
echo "Checking service health..."
curl -f http://localhost:8000/health || echo "API health check failed"
curl -f http://localhost:8080 || echo "Web dashboard check failed"

echo "✅ Deployment complete!"
echo ""
echo "Services running:"
echo "- API: http://localhost:8000"
echo "- Web Dashboard: http://localhost:8080"
echo "- Docs: http://localhost:8000/docs"
echo ""
