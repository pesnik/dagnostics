#!/bin/bash
# CPU-only training startup script for servers without GPU
# Fallback option when NVIDIA Docker is not available

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="dagnostics-training-server-cpu"
COMPOSE_FILE="docker/docker-compose.training.cpu.yml"
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

# Function to determine docker compose command
get_compose_cmd() {
    if command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    elif docker compose version &> /dev/null 2>&1; then
        echo "docker compose"
    else
        print_error "Neither 'docker-compose' nor 'docker compose' is available"
        exit 1
    fi
}

# Function to check prerequisites (CPU version)
check_prerequisites() {
    print_status "Checking prerequisites for CPU training..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi

    # Check Docker Compose
    COMPOSE_CMD=$(get_compose_cmd)
    print_success "Using compose command: $COMPOSE_CMD"

    # Check available memory (CPU training needs more RAM)
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $TOTAL_MEM -lt 8 ]]; then
        print_warning "CPU training recommended: 8GB+ RAM (detected: ${TOTAL_MEM}GB)"
        print_warning "Training may be slow or fail with insufficient memory"
    else
        print_success "Memory check passed: ${TOTAL_MEM}GB available"
    fi

    # No GPU check for CPU version
    print_success "CPU training prerequisites check passed"
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

# Function to start CPU training server
start_server() {
    print_status "Starting DAGnostics CPU training server..."

    COMPOSE_CMD=$(get_compose_cmd)

    # Stop existing container if running
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        print_warning "Stopping existing container..."
        $COMPOSE_CMD -f "$COMPOSE_FILE" down
    fi

    # Build and start CPU version
    print_status "Building CPU training container..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up --build -d

    # Wait for server to be ready
    print_status "Waiting for server to be ready..."
    for i in {1..60}; do  # Longer timeout for CPU
        if curl -s -f http://localhost:8001/health > /dev/null 2>&1; then
            print_success "CPU training server is ready!"
            break
        fi

        if [[ $i -eq 60 ]]; then
            print_error "Server failed to start within 60 seconds"
            $COMPOSE_CMD -f "$COMPOSE_FILE" logs
            exit 1
        fi

        sleep 1
    done
}

# Function to show server info
show_server_info() {
    COMPOSE_CMD=$(get_compose_cmd)

    print_success "🚀 DAGnostics CPU Training Server Started!"
    echo
    echo "📍 Server URL: http://localhost:8001"
    echo "📚 API Documentation: http://localhost:8001/docs"
    echo "❤️  Health Check: http://localhost:8001/health"
    echo
    echo "⚠️  CPU Training Notes:"
    echo "  - Training will be slower than GPU (10-30 minutes vs 2-5 minutes)"
    echo "  - Use small batch sizes (batch_size=1)"
    echo "  - Use fewer epochs for testing (epochs=1-3)"
    echo "  - Quantization is automatically disabled"
    echo
    echo "📊 Container Status:"
    $COMPOSE_CMD -f "$COMPOSE_FILE" ps
    echo
    echo "🔧 Useful Commands:"
    echo "  View logs:     $COMPOSE_CMD -f $COMPOSE_FILE logs -f"
    echo "  Stop server:   $COMPOSE_CMD -f $COMPOSE_FILE down"
    echo "  Restart:       $COMPOSE_CMD -f $COMPOSE_FILE restart"
    echo "  Shell access:  docker exec -it $CONTAINER_NAME bash"
    echo
    echo "🎯 Start CPU Training:"
    echo "  Quick test:    just train-cpu-test"
    echo "  Local CPU:     just train-cpu"
    echo "  Remote CPU:    dagnostics training remote-train --server-url http://localhost:8001 --epochs 1 --batch-size 1"
    echo
}

# Function to test server
test_server() {
    print_status "Testing CPU server connectivity..."

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

    # Check CPU info
    CPU_INFO=$(docker exec "$CONTAINER_NAME" nproc 2>/dev/null || echo "Unknown")
    print_success "CPU cores available: $CPU_INFO"

    # Check if CPU mode is enabled
    CPU_MODE=$(docker exec "$CONTAINER_NAME" printenv DAGNOSTICS_FORCE_CPU 2>/dev/null || echo "false")
    print_success "CPU mode enabled: $CPU_MODE"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  start     Start the CPU training server (default)"
    echo "  stop      Stop the CPU training server"
    echo "  restart   Restart the CPU training server"
    echo "  status    Show server status"
    echo "  logs      Show server logs"
    echo "  test      Test server connectivity"
    echo "  shell     Open shell in container"
    echo "  clean     Clean up containers and images"
    echo "  help      Show this help message"
    echo
    echo "CPU Training Notes:"
    echo "  - No GPU required, uses CPU-only PyTorch"
    echo "  - Slower training but works on any machine"
    echo "  - Ideal for testing and small datasets"
    echo "  - Automatically optimizes settings for CPU"
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
        COMPOSE_CMD=$(get_compose_cmd)
        print_status "Stopping CPU training server..."
        $COMPOSE_CMD -f "$COMPOSE_FILE" down
        print_success "Server stopped"
        ;;

    restart)
        COMPOSE_CMD=$(get_compose_cmd)
        print_status "Restarting CPU training server..."
        $COMPOSE_CMD -f "$COMPOSE_FILE" restart
        sleep 5
        test_server
        print_success "Server restarted"
        ;;

    status)
        COMPOSE_CMD=$(get_compose_cmd)
        echo "📊 Container Status:"
        $COMPOSE_CMD -f "$COMPOSE_FILE" ps
        echo
        if curl -s -f http://localhost:8001/health > /dev/null 2>&1; then
            print_success "CPU server is running and healthy"
            test_server
        else
            print_warning "Server is not responding"
        fi
        ;;

    logs)
        COMPOSE_CMD=$(get_compose_cmd)
        $COMPOSE_CMD -f "$COMPOSE_FILE" logs -f
        ;;

    test)
        test_server
        ;;

    shell)
        docker exec -it "$CONTAINER_NAME" bash
        ;;

    clean)
        COMPOSE_CMD=$(get_compose_cmd)
        print_warning "This will remove all containers and images. Continue? (y/N)"
        read -r CONFIRM
        if [[ $CONFIRM =~ ^[Yy]$ ]]; then
            $COMPOSE_CMD -f "$COMPOSE_FILE" down --rmi all --volumes
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
