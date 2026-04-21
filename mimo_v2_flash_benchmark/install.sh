#!/bin/bash
set -e

echo "=== Installing on $(hostname) ==="

# Install JAX with TPU support (matching report versions: jax 0.8.1, libtpu 0.0.30)
pip install -U "jax[tpu]==0.8.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>&1 | tail -5

# Install flax
pip install flax 2>&1 | tail -3

# Clone sglang-jax if not already present
if [ ! -d /sglang-jax ]; then
    sudo git clone -b epic/mimo-v2-flash https://github.com/sgl-project/sglang-jax.git /sglang-jax
    sudo chown -R $(whoami) /sglang-jax
fi

# Install sglang-jax
cd /sglang-jax/python
pip install -e . 2>&1 | tail -5

# Install evalscope for accuracy testing
pip install evalscope==0.17.1 2>&1 | tail -3

# Install huggingface_hub for model download
pip install huggingface_hub 2>&1 | tail -3

# Verify installation
echo "=== Verification ==="
pip list 2>/dev/null | egrep 'jax|flax|libtpu|sglang|evalscope'
python3 -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')"
echo "=== Done on $(hostname) ==="
