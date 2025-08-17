# Docker Deployment Guide

This directory contains Docker configurations for different deployment scenarios.

## Available Configurations

### 1. GPU Training (Production)
- **File**: `docker-compose.training.yml`
- **Dockerfile**: `Dockerfile.training`
- **Use case**: GPU-accelerated fine-tuning on production servers
- **Requirements**: NVIDIA GPU with Docker GPU support

```bash
# Start GPU training server
docker-compose -f docker/docker-compose.training.yml up

# Access at: http://localhost:8001
```

### 2. CPU Training (Fallback)
- **File**: `docker-compose.training.cpu.yml`
- **Dockerfile**: `Dockerfile.training.cpu`
- **Use case**: CPU-only training when GPU is not available
- **Requirements**: 8GB+ RAM, no GPU needed

```bash
# Start CPU training server
docker-compose -f docker/docker-compose.training.cpu.yml up

# Access at: http://localhost:8001
```

### 3. Legacy Training
- **File**: `../Dockerfile.training`
- **Use case**: Simple training container without server
- **Requirements**: Basic CPU/GPU setup

```bash
# Build and run legacy container
docker build -f Dockerfile.training -t dagnostics-training .
docker run -it dagnostics-training
```

## Environment Variables

### GPU Configuration
```bash
# GPU settings (production)
HF_HOME=/app/huggingface_cache
TRANSFORMERS_CACHE=/app/huggingface_cache
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### CPU Configuration
```bash
# CPU settings (fallback)
DAGNOSTICS_FORCE_CPU=true
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=""  # Hide GPU devices
```

## Usage Examples

### Quick CPU Test
```bash
# Test CPU training locally first
python test_cpu_training.py

# Then run CPU Docker
docker-compose -f docker/docker-compose.training.cpu.yml up
```

### Force CPU Mode on Existing Setup
```bash
# Override GPU setup to use CPU
DAGNOSTICS_FORCE_CPU=true docker-compose -f docker/docker-compose.training.yml up
```

### Resource Limits
```bash
# CPU container automatically sets:
# - CPU limit: 4 cores
# - Memory limit: 8GB
# - Shared memory: 1GB

# GPU container uses:
# - All available GPUs
# - Shared memory: 2GB
# - No CPU/memory limits
```

## Training Commands in Docker

Once the server is running, submit training jobs:

```bash
# Submit CPU training job
curl -X POST http://localhost:8001/training/submit \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-cpu-job",
    "model_name": "microsoft/DialoGPT-small",
    "epochs": 1,
    "batch_size": 1,
    "use_quantization": false
  }'

# Check job status
curl http://localhost:8001/training/status/test-cpu-job

# Download completed model
curl -O http://localhost:8001/training/download/test-cpu-job
```

## Troubleshooting

### Container Won't Start
```bash
# Check Docker daemon
sudo systemctl status docker

# Check GPU support (for GPU containers)
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Check logs
docker-compose logs dagnostics-trainer-cpu
```

### Out of Memory
```bash
# Reduce resource limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G  # Reduce from 8G
      cpus: '2.0'  # Reduce from 4.0
```

### Training Dependencies Missing
```bash
# Rebuild with training dependencies
docker-compose build --no-cache

# Check if ML libraries are installed
docker-compose exec dagnostics-trainer-cpu python -c "import torch; print(torch.__version__)"
```

## Development Workflow

### Local Development
```bash
# 1. Test locally first
just train-cpu-test

# 2. Build and test container
docker-compose -f docker/docker-compose.training.cpu.yml build
docker-compose -f docker/docker-compose.training.cpu.yml up

# 3. Submit test job
curl -X POST http://localhost:8001/training/submit -H "Content-Type: application/json" -d '{"job_id": "dev-test", "model_name": "microsoft/DialoGPT-small", "epochs": 1, "batch_size": 1}'
```

### Production Deployment
```bash
# 1. Use GPU container for production
docker-compose -f docker/docker-compose.training.yml up -d

# 2. Monitor with logs
docker-compose logs -f dagnostics-trainer

# 3. Health check
curl http://localhost:8001/health
```

## Next Steps

1. **For Testing**: Use CPU containers to verify the workflow works
2. **For Production**: Use GPU containers for faster training
3. **For Development**: Use local Python environment with `just train-cpu`

See the main [Fine-Tuning Guide](../docs/fine_tuning_guide.md) for detailed usage instructions.
