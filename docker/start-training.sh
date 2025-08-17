#!/bin/bash
# Production-ready training startup script for remote servers
# Optimized for HuggingFace compatibility and Ollama deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="dagnostics-training-server"
COMPOSE_FILE="docker/docker-compose.training.yml"
DATA_DIR="./data"
CACHE_DIR="./cache"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check NVIDIA Docker
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        print_error "NVIDIA Docker runtime not available"
        print_error "Install with: sudo apt install nvidia-docker2"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to setup directories
setup_directories() {
    print_status "Setting up directories..."
    
    # Create required directories
    mkdir -p "$DATA_DIR/fine_tuning"
    mkdir -p "$CACHE_DIR/huggingface"
    mkdir -p "$CACHE_DIR/torch"
    mkdir -p "./server_data/models"
    mkdir -p "./server_data/datasets"
    mkdir -p "./evaluations"
    
    # Set proper permissions
    chmod -R 755 "$DATA_DIR"
    chmod -R 755 "$CACHE_DIR"
    chmod -R 755 "./server_data"
    chmod -R 755 "./evaluations"
    
    print_success "Directories setup complete"
}

# Function to check training data
check_training_data() {
    print_status "Checking training data..."
    
    if [[ -f "$DATA_DIR/fine_tuning/train_dataset.jsonl" ]]; then
        TRAIN_COUNT=$(wc -l < "$DATA_DIR/fine_tuning/train_dataset.jsonl")
        print_success "Training dataset found: $TRAIN_COUNT examples"
    else
        print_warning "Training dataset not found at $DATA_DIR/fine_tuning/train_dataset.jsonl"
        print_warning "Run 'python scripts/prepare_training_data.py' first"
    fi
    
    if [[ -f "$DATA_DIR/fine_tuning/validation_dataset.jsonl" ]]; then
        VAL_COUNT=$(wc -l < "$DATA_DIR/fine_tuning/validation_dataset.jsonl")
        print_success "Validation dataset found: $VAL_COUNT examples"
    else
        print_warning "Validation dataset not found"
    fi
}

# Function to start training server
start_server() {
    print_status "Starting DAGnostics training server..."
    
    # Stop existing container if running
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        print_warning "Stopping existing container..."
        docker-compose -f "$COMPOSE_FILE" down
    fi
    
    # Build and start
    docker-compose -f "$COMPOSE_FILE" up --build -d
    
    # Wait for server to be ready
    print_status "Waiting for server to be ready..."
    for i in {1..30}; do
        if curl -s -f http://localhost:8001/health > /dev/null 2>&1; then
            print_success "Training server is ready!"
            break
        fi
        
        if [[ $i -eq 30 ]]; then
            print_error "Server failed to start within 30 seconds"
            docker-compose -f "$COMPOSE_FILE" logs
            exit 1
        fi
        
        sleep 1
    done
}

# Function to show server info
show_server_info() {
    print_success "🚀 DAGnostics Training Server Started!"
    echo
    echo "📍 Server URL: http://localhost:8001"
    echo "📚 API Documentation: http://localhost:8001/docs"
    echo "❤️  Health Check: http://localhost:8001/health"
    echo
    echo "📊 Container Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    echo
    echo "🔧 Useful Commands:"
    echo "  View logs:     docker-compose -f $COMPOSE_FILE logs -f"
    echo "  Stop server:   docker-compose -f $COMPOSE_FILE down"
    echo "  Restart:       docker-compose -f $COMPOSE_FILE restart"
    echo "  Shell access:  docker exec -it $CONTAINER_NAME bash"
    echo
    echo "🎯 Start Training:"
    echo "  Local:   just train-remote http://localhost:8001"
    echo "  Manual:  dagnostics training remote-train --server-url http://localhost:8001"
    echo
}

# Function to test server
test_server() {
    print_status "Testing server connectivity..."
    
    # Test health endpoint
    if curl -s -f http://localhost:8001/health > /dev/null; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
        return 1
    fi
    
    # Test server info
    SERVER_INFO=$(curl -s http://localhost:8001/ | jq -r '.service // "Unknown"' 2>/dev/null || echo "Unknown")
    print_success "Server info: $SERVER_INFO"
    
    # Check GPU availability
    GPU_INFO=$(docker exec "$CONTAINER_NAME" nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "No GPU detected")
    print_success "GPU: $GPU_INFO"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  start     Start the training server (default)"
    echo "  stop      Stop the training server"
    echo "  restart   Restart the training server"
    echo "  status    Show server status"
    echo "  logs      Show server logs"
    echo "  test      Test server connectivity"
    echo "  shell     Open shell in container"
    echo "  clean     Clean up containers and images"
    echo "  help      Show this help message"
}

# Main script logic
case "${1:-start}" in
    start)
        check_prerequisites
        setup_directories
        check_training_data
        start_server
        test_server
        show_server_info
        ;;
    
    stop)
        print_status "Stopping training server..."
        docker-compose -f "$COMPOSE_FILE" down
        print_success "Server stopped"
        ;;
    
    restart)
        print_status "Restarting training server..."
        docker-compose -f "$COMPOSE_FILE" restart
        sleep 5
        test_server
        print_success "Server restarted"
        ;;
    
    status)
        echo "📊 Container Status:"
        docker-compose -f "$COMPOSE_FILE" ps
        echo
        if curl -s -f http://localhost:8001/health > /dev/null 2>&1; then
            print_success "Server is running and healthy"
            test_server
        else
            print_warning "Server is not responding"
        fi
        ;;
    
    logs)
        docker-compose -f "$COMPOSE_FILE" logs -f
        ;;
    
    test)
        test_server
        ;;
    
    shell)
        docker exec -it "$CONTAINER_NAME" bash
        ;;
    
    clean)
        print_warning "This will remove all containers and images. Continue? (y/N)"
        read -r CONFIRM
        if [[ $CONFIRM =~ ^[Yy]$ ]]; then
            docker-compose -f "$COMPOSE_FILE" down --rmi all --volumes
            docker system prune -f
            print_success "Cleanup complete"
        fi
        ;;
    
    help|--help|-h)
        show_usage
        ;;
    
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac