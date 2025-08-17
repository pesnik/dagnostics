#!/usr/bin/env python3
"""
Remote Training Setup Script

Helps set up remote training infrastructure with various cloud providers
and local GPU machines.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_step(step, description):
    """Print a formatted step"""
    print(f"\n{step}. {description}")
    print("-" * (len(str(step)) + len(description) + 2))


def check_dependencies():
    """Check if required tools are installed"""
    tools = {
        "docker": "Docker for containerized training",
        "ssh": "SSH for remote server access",
        "scp": "SCP for file transfer",
    }

    missing = []
    for tool, description in tools.items():
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            missing.append(f"  ❌ {tool} - {description}")
        else:
            print(f"  ✅ {tool} - {description}")

    if missing:
        print("\nMissing tools:")
        for tool in missing:
            print(tool)
        return False

    return True


def setup_local_gpu():
    """Setup local GPU machine as training server"""
    print_section("Setting Up Local GPU Machine")

    print(
        """
This will help you set up a local GPU machine as a training server.

Requirements:
- Separate machine with NVIDIA GPU
- CUDA drivers installed
- Network connectivity between machines
"""
    )

    gpu_ip = input("Enter GPU machine IP address: ")
    gpu_user = input("Enter GPU machine username: ")

    print_step(1, "Test SSH connection")
    ssh_cmd = f"ssh {gpu_user}@{gpu_ip} 'echo Connection successful'"
    print(f"Running: {ssh_cmd}")

    result = subprocess.run(ssh_cmd, shell=True)
    if result.returncode != 0:
        print("❌ SSH connection failed. Please check:")
        print("  - IP address and username")
        print("  - SSH key setup")
        print("  - Network connectivity")
        return False

    print_step(2, "Copy codebase to GPU machine")
    print("Copying current directory to GPU machine...")

    scp_cmd = f"scp -r . {gpu_user}@{gpu_ip}:~/dagnostics/"
    subprocess.run(scp_cmd, shell=True)

    print_step(3, "Install dependencies on GPU machine")
    install_cmd = f"""ssh {gpu_user}@{gpu_ip} '
    cd ~/dagnostics &&
    pip install -e .[finetuning] &&
    echo "Dependencies installed successfully"
    '"""

    result = subprocess.run(install_cmd, shell=True)
    if result.returncode != 0:
        print("❌ Dependency installation failed")
        return False

    print_step(4, "Start training server")
    server_cmd = f"""ssh {gpu_user}@{gpu_ip} '
    cd ~/dagnostics &&
    nohup python -m dagnostics.training.training_server --host 0.0.0.0 --port 8001 > training_server.log 2>&1 &
    echo "Training server started on port 8001"
    '"""

    subprocess.run(server_cmd, shell=True)

    print_step(5, "Test server connection")
    test_url = f"http://{gpu_ip}:8001"

    print(
        f"""
✅ Local GPU training server setup complete!

🔗 Server URL: {test_url}
📋 API Docs: {test_url}/docs
📊 Status: {test_url}/health

Test the connection:
    curl {test_url}/health

Start training from your main machine:
    just train-remote {test_url} microsoft/DialoGPT-small 3

Or with dagnostics CLI:
    dagnostics training remote-train --server-url {test_url}
"""
    )

    return True


def setup_docker_training():
    """Setup Docker-based training"""
    print_section("Setting Up Docker Training Server")

    print(
        """
This will create a Docker container for training with GPU support.

Requirements:
- Docker installed
- NVIDIA Docker runtime (nvidia-docker2)
- NVIDIA GPU
"""
    )

    # Check Docker
    result = subprocess.run(["docker", "--version"], capture_output=True)
    if result.returncode != 0:
        print("❌ Docker not found. Please install Docker first.")
        return False

    print("✅ Docker found")

    # Check NVIDIA Docker
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "nvidia/cuda:11.8-base-ubuntu22.04",
            "nvidia-smi",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("❌ NVIDIA Docker runtime not available")
        print("Install with:")
        print("  # Add NVIDIA package repositories")
        print(
            "  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
        )
        print(
            "  curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
        )
        print("  sudo apt-get update && sudo apt-get install -y nvidia-docker2")
        print("  sudo systemctl restart docker")
        return False

    print("✅ NVIDIA Docker runtime available")

    print_step(1, "Create Dockerfile for training")

    dockerfile_content = """FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app/

# Install Python dependencies
RUN pip3 install -e .[finetuning]

# Expose port
EXPOSE 8001

# Create data directories
RUN mkdir -p server_data/uploads server_data/models server_data/datasets

# Start training server
CMD ["python3", "-m", "dagnostics.training.training_server", "--host", "0.0.0.0", "--port", "8001"]
"""

    with open("Dockerfile.training", "w") as f:
        f.write(dockerfile_content)

    print("✅ Dockerfile.training created")

    print_step(2, "Build Docker image")
    build_cmd = [
        "docker",
        "build",
        "-f",
        "Dockerfile.training",
        "-t",
        "dagnostics-trainer",
        ".",
    ]
    print(f"Running: {' '.join(build_cmd)}")

    result = subprocess.run(build_cmd)
    if result.returncode != 0:
        print("❌ Docker build failed")
        return False

    print("✅ Docker image built: dagnostics-trainer")

    print_step(3, "Start training container")
    run_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        "dagnostics-training-server",
        "--gpus",
        "all",
        "-p",
        "8001:8001",
        "-v",
        f"{Path.cwd()}/server_data:/app/server_data",
        "dagnostics-trainer",
    ]

    print(f"Running: {' '.join(run_cmd)}")
    result = subprocess.run(run_cmd)

    if result.returncode != 0:
        print("❌ Failed to start container")
        return False

    print("✅ Training server container started")

    print_step(4, "Test server")
    import time

    print("Waiting 10 seconds for server to start...")
    time.sleep(10)

    test_result = subprocess.run(
        ["curl", "-s", "http://localhost:8001/health"], capture_output=True, text=True
    )

    if test_result.returncode == 0:
        print("✅ Server is responding")
        print(f"Response: {test_result.stdout}")
    else:
        print("⚠️ Server might still be starting up")

    print(
        f"""
✅ Docker training server setup complete!

🔗 Server URL: http://localhost:8001
📋 API Docs: http://localhost:8001/docs
📊 Status: http://localhost:8001/health

Container management:
    docker logs dagnostics-training-server     # View logs
    docker stop dagnostics-training-server     # Stop server
    docker start dagnostics-training-server    # Start server
    docker rm dagnostics-training-server       # Remove container

Start training:
    just train-remote http://localhost:8001 microsoft/DialoGPT-small 3
"""
    )

    return True


def setup_cloud_instance():
    """Setup cloud instance for training"""
    print_section("Setting Up Cloud Training Instance")

    print(
        """
This will help you set up a cloud GPU instance for training.

Supported providers:
1. AWS (g4dn, p3, p4 instances)
2. Google Cloud (T4, V100, A100 instances)
3. Azure (NC, ND, NV series)

You'll need to:
1. Launch a GPU instance manually
2. Run this script to configure it
"""
    )

    provider = input("Cloud provider (aws/gcp/azure): ").lower()
    instance_ip = input("Instance public IP: ")
    key_path = input("SSH key path: ")
    username = input("Username (ubuntu/centos/etc): ")

    print_step(1, "Test connection")
    ssh_cmd = f"ssh -i {key_path} {username}@{instance_ip} 'echo Connection successful'"

    result = subprocess.run(ssh_cmd, shell=True)
    if result.returncode != 0:
        print("❌ SSH connection failed")
        return False

    print_step(2, "Install system dependencies")
    if provider == "aws":
        setup_cmd = f"""ssh -i {key_path} {username}@{instance_ip} '
        sudo apt update &&
        sudo apt install -y python3 python3-pip git &&
        pip3 install --upgrade pip
        '"""
    elif provider == "gcp":
        setup_cmd = f"""ssh -i {key_path} {username}@{instance_ip} '
        sudo apt update &&
        sudo apt install -y python3 python3-pip git &&
        pip3 install --upgrade pip
        '"""
    else:  # Azure or generic
        setup_cmd = f"""ssh -i {key_path} {username}@{instance_ip} '
        sudo apt update &&
        sudo apt install -y python3 python3-pip git &&
        pip3 install --upgrade pip
        '"""

    subprocess.run(setup_cmd, shell=True)

    print_step(3, "Transfer codebase")
    scp_cmd = f"scp -i {key_path} -r . {username}@{instance_ip}:~/dagnostics/"
    subprocess.run(scp_cmd, shell=True)

    print_step(4, "Install training dependencies")
    install_cmd = f"""ssh -i {key_path} {username}@{instance_ip} '
    cd ~/dagnostics &&
    pip3 install -e .[finetuning]
    '"""

    subprocess.run(install_cmd, shell=True)

    print_step(5, "Start training server")
    server_cmd = f"""ssh -i {key_path} {username}@{instance_ip} '
    cd ~/dagnostics &&
    nohup python3 -m dagnostics.training.training_server --host 0.0.0.0 --port 8001 > training_server.log 2>&1 &
    echo "Server started"
    '"""

    subprocess.run(server_cmd, shell=True)

    print(
        f"""
✅ Cloud training server setup complete!

🔗 Server URL: http://{instance_ip}:8001
📋 API Docs: http://{instance_ip}:8001/docs

⚠️  Security Note: Make sure port 8001 is open in your security group/firewall

Start training:
    just train-remote http://{instance_ip}:8001 microsoft/DialoGPT-small 3
"""
    )

    return True


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup remote training infrastructure")
    parser.add_argument(
        "--mode", choices=["local", "docker", "cloud"], help="Setup mode"
    )

    args = parser.parse_args()

    print_section("DAGnostics Remote Training Setup")
    print("This script helps you set up remote training infrastructure")

    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return 1

    if not args.mode:
        print("\nAvailable setup modes:")
        print("1. local  - Use another local machine with GPU")
        print("2. docker - Use Docker container on current machine")
        print("3. cloud  - Use cloud GPU instance (AWS/GCP/Azure)")

        mode = input("\nSelect mode (local/docker/cloud): ").lower()
    else:
        mode = args.mode

    if mode == "local":
        success = setup_local_gpu()
    elif mode == "docker":
        success = setup_docker_training()
    elif mode == "cloud":
        success = setup_cloud_instance()
    else:
        print("❌ Invalid mode selected")
        return 1

    if success:
        print("\n🎉 Remote training setup completed successfully!")
        print("\nNext steps:")
        print("1. Test the server connection")
        print("2. Run: just training-status")
        print("3. Start training: just train-remote")
        return 0
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
