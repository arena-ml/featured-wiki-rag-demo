# Base Image
FROM nvidia/cuda:12.8.0-base-ubuntu24.04

# Set environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1 \
    PIP_NO_CACHE_DIR=1 \
    FORCE_CMAKE=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_PYPI_URL=https://pypi.org/simple
    TZ=UTC


# Copy and install Python dependencies
COPY requirements.txt requirements.txt

# Install dependencies and immediately clean up to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp-dev \
    python3 \
    python3-pip \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    python3-dev \
    ninja-build \
    libclblast-dev \
    libopenblas-dev \
    libomp-dev \
    libgomp1 \
    wget \
    git \
    curl && \
    pip3 install --debug --no-cache-dir --verbose --break-system-packages --index-url https://download.pytorch.org/whl/cu126 torch && \
    CMAKE_ARGS="-DGGML_NATIVE=OFF -DCMAKE_CXX_FLAGS='-march=native' -DCMAKE_C_FLAGS='-march=native' -DGGML_CPU_ARM_ARCH=native" pip3 install --debug --no-cache-dir --verbose --break-system-packages \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cuda125 llama-cpp-python && \
    pip3 install --debug --no-cache-dir --verbose --upgrade --break-system-packages --index-url https://pypi.org/simple -r requirements.txt && \
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /root/.cache /tmp/* /var/tmp/* /var/lib/apt/lists/*

# Create model directory and download model
RUN mkdir -p /app/jinv3/modelCache

# Copy script and download model inside a single RUN layer
COPY saveModels.py saveModels.py
RUN python3 saveModels.py && \
    curl -o /app/Phi-3.5-mini-instruct-Q6_K.gguf -L \
    "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q6_K.gguf" || \
    { echo "Failed to download LLM model" && exit 1; } && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /root/.cache/* /tmp/* /var/tmp/* /var/lib/apt/lists/*

