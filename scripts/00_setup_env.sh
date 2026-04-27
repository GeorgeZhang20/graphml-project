#!/usr/bin/env bash
# One-shot environment setup. Run from the project root.
#
# Usage:
#   bash scripts/00_setup_env.sh
#
# Creates a conda env called `tape-proj` with Python 3.10 and installs deps.

set -euo pipefail

ENV_NAME="${ENV_NAME:-tape-proj}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda first."
  exit 1
fi

# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Env '$ENV_NAME' already exists. Activating."
else
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

# CPU-only torch (Intel Mac max). On Colab/Linux+CUDA, install torch separately.
pip install --upgrade pip
pip install torch==2.2.2 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

python - <<'PY'
import torch, torch_geometric, ogb, transformers
print(f"OK: torch={torch.__version__} pyg={torch_geometric.__version__} "
      f"ogb={ogb.__version__} transformers={transformers.__version__}")
PY
