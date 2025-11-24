# Use official vLLM base image for proper GPU support
# This image already has CUDA 12.1, PyTorch, and vLLM properly configured
FROM vllm/vllm-openai:latest

# Install additional dependencies for your custom FastAPI wrapper
WORKDIR /app

# Copy requirements (excluding vLLM since it's already installed)
COPY requirements.txt .

# Install only the additional packages (FastAPI, etc.)
# vLLM and PyTorch are already installed in the base image
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    pydantic>=2.0.0 \
    pydantic-settings>=2.0.0 \
    openai>=1.0.0 \
    python-dotenv>=1.0.0 \
    httpx>=0.25.0 \
    aiofiles>=23.0.0 \
    click>=8.0.0 \
    huggingface-hub>=0.19.0 \
    prometheus-client>=0.18.0 \
    python-json-logger>=2.0.0

# Copy application code
COPY config ./config
COPY src ./src

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Override the vLLM entrypoint to run your custom FastAPI application
ENTRYPOINT ["python3", "-m", "src.server"]
