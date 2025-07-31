FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ARG MODEL_NAME=hf.co/unsloth/SmolLM3-3B-128K-GGUF:BF16
# Set environment variables for optimization
# adding ven in path activates it automatics at start of image instance
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    TZ=UTC \
    MODEL_NAME=${MODEL_NAME} \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Copy Python dependencies list
COPY requirements.txt requirements.txt
COPY devOps/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Install dependencies and immediately clean up in one layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    wget \
    git \
    lshw \
    curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.cargo/bin:$PATH" && \
    uv venv --system-site-packages /opt/venv && \
    uv pip install --no-cache -r requirements.txt && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    (ollama serve > /dev/null 2>&1 &) && \
    sleep 15 && \
    ollama pull ${MODEL_NAME} && \
    apt-get autoremove --purge  -y && \
    apt-get clean -y && \
    rm -rf /root/.cache/* /tmp/* /var/tmp/* /var/lib/apt/lists/*

ENTRYPOINT ["/entrypoint.sh"]