FROM rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0

# Install system dependencies and add deadsnakes PPA for Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Update alternatives to use Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --set python3 /usr/bin/python3.11

# Critical environment variables for RDNA3 stability and memory tuning
ENV TORCH_BLAS_PREFER_HIPBLASLT=0
ENV PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6
ENV PYTORCH_ALLOC_CONF=max_split_size_mb:128
ENV HSA_XNACK=1
ENV ROCM_ALLOW_HIDDEN_KERNEL_FLAGS=1

# Using gfx1100 for RX 7900 XT
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
ENV PYTORCH_ROCM_ARCH=gfx1100
ENV HIP_VISIBLE_DEVICES=0,1

# Append ROCm paths to PATH and LD_LIBRARY_PATH
ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Optional: Install Node.js if needed (comment out if not required)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g @anthropic-ai/claude-code

# Verify PyTorch installation
RUN python3 -c "\
import torch; \
print('PyTorch version:', torch.__version__); \
print('ROCm version:', torch.version.hip); \
print('CUDA/ROCm available:', torch.cuda.is_available()); \
print('Device count:', torch.cuda.device_count())"

# Final default command
CMD ["/bin/bash"]
