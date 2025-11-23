#!/bin/bash

# Start script for vLLM API server
# Usage: ./scripts/start_server.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  vLLM API Server Startup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo -e "Creating .env from .env.example..."
    cp .env.example .env
    echo -e "${GREEN}Created .env file. Please configure it with your settings.${NC}"
    echo ""
fi

# Load environment variables
if [ -f .env ]; then
    echo -e "${BLUE}Loading environment variables from .env${NC}"
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Not running in a virtual environment${NC}"
    echo -e "It's recommended to use a virtual environment."
    echo ""
fi

# Check if dependencies are installed
echo -e "${BLUE}Checking dependencies...${NC}"
if ! python -c "import vllm" 2>/dev/null; then
    echo -e "${RED}Error: vLLM not installed${NC}"
    echo -e "Please install dependencies: pip install -r requirements.txt"
    exit 1
fi

if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${RED}Error: FastAPI not installed${NC}"
    echo -e "Please install dependencies: pip install -r requirements.txt"
    exit 1
fi

echo -e "${GREEN}Dependencies OK${NC}"
echo ""

# Display configuration
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Host: ${HOST:-0.0.0.0}"
echo -e "  Port: ${PORT:-8000}"
echo -e "  Model: ${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
echo -e "  GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION:-0.9}"
echo -e "  Tensor Parallel Size: ${TENSOR_PARALLEL_SIZE:-1}"
echo ""

# Check if GPU is available
echo -e "${BLUE}Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU may not be available.${NC}"
    echo -e "vLLM will fall back to CPU (very slow)."
    echo ""
fi

# Start the server
echo -e "${GREEN}Starting vLLM API server...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

python -m src.server
