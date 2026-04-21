#!/bin/bash
set -ex

echo "=== Installing on $(hostname) ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

# Clone sglang-jax if not present
if [ ! -d /sglang-jax ]; then
    sudo git clone -b epic/mimo-v2-flash https://github.com/sgl-project/sglang-jax.git /sglang-jax
    sudo chown -R $(whoami) /sglang-jax
fi

# Create venv with Python 3.12
cd /sglang-jax
uv venv --python 3.12
source .venv/bin/activate

# Install sglang-jax with TPU support
uv pip install -e "python[tpu]"

# Install evalscope for accuracy testing
uv pip install evalscope==0.17.1 || uv pip install evalscope

# Verify installation
echo "=== Verification ==="
pip list 2>/dev/null | egrep 'jax|flax|libtpu|sglang|evalscope'
python3 -c "import jax; print(f'JAX version: {jax.__version__}')"
echo "=== Done on $(hostname) ==="
