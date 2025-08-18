#!/bin/bash
# Setup script for Docker permission issues
# Run this before starting Docker containers

set -e

echo "🔧 Setting up Docker permissions for DAGnostics Training..."

# Get current user ID and group ID
export USER_ID=$(id -u)
export GROUP_ID=$(id -g)

echo "📋 Host User ID: $USER_ID"
echo "📋 Host Group ID: $GROUP_ID"

# Create directories if they don't exist
echo "📁 Creating necessary directories..."
mkdir -p \
    server_data/models \
    server_data/datasets \
    server_data/uploads \
    cache/huggingface \
    cache/torch \
    data \
    evaluations \
    config

# Set proper ownership for mounted volumes
echo "🔑 Setting directory ownership..."
sudo chown -R $USER_ID:$GROUP_ID \
    server_data \
    cache \
    data \
    evaluations \
    config 2>/dev/null || {
    echo "⚠️  Could not set ownership with sudo. Setting permissions without sudo..."
    chmod -R 755 \
        server_data \
        cache \
        data \
        evaluations \
        config
}

# Create .env file with user IDs
echo "📝 Creating .env file with user mapping..."
cat > .env << EOF
# User mapping for Docker containers
USER_ID=$USER_ID
GROUP_ID=$GROUP_ID

# Optional: HuggingFace token for private models
# HF_TOKEN=your_token_here
EOF

echo "✅ Setup complete! You can now run:"
echo "   docker-compose -f docker-compose.training.yml up"
echo "   or"
echo "   docker-compose -f docker-compose.training.cpu.yml up"
echo ""
echo "💡 If you still get permission errors, run this script again with sudo."
