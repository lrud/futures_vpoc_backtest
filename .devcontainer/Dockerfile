FROM rocm/pytorch:rocm6.3.2_ubuntu24.04_py3.12_pytorch_release_2.4.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    rocm-smi \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Upgrade pip and setuptools without caching
RUN pip3 install --no-cache-dir --upgrade pip setuptools

# Using gfx1100 for RX 7900 XT
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0
# Correct architecture for RX 7900 XT
ENV PYTORCH_ROCM_ARCH=gfx1100
ENV HIP_VISIBLE_DEVICES=0,1
ENV PATH="/opt/rocm/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"

# Verify PyTorch installation
RUN python3 -c "\
import torch; \
print('PyTorch version:', torch.__version__); \
print('ROCm version:', torch.version.hip); \
print('CUDA/ROCm available:', torch.cuda.is_available()); \
print('Device count:', torch.cuda.device_count())"

# Note: requirements.txt will be mounted from the host
# and installed via postCreateCommand in devcontainer.json

# Set default command
CMD ["/bin/bash"]