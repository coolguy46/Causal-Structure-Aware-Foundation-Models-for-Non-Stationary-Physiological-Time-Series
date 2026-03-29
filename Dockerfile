FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    git wget curl htop tmux \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attention (optional, may fail on some architectures)
RUN pip install flash-attn --no-build-isolation 2>/dev/null || true

# Copy project
COPY . /workspace/causal-biosignal-fm/
WORKDIR /workspace/causal-biosignal-fm

# Create data directories
RUN mkdir -p /workspace/data/{sleep_edf,chbmit,ptbxl} \
    /workspace/outputs \
    /workspace/checkpoints

# Default command
CMD ["python", "-m", "src.train"]
