#!/bin/bash
#
# Installer for the fmfwi (FM-FWI: foundation-model objective for FWI) env
#
# Run: ./install_env.sh
#
set -euo pipefail

echo 'Creating fmfwi environment'

# create conda env (numpy/scipy/matplotlib/jupyter only; see environment.yml
# for why torch and the rest of the pip stack are deferred to this script
# instead of living in environment.yml's pip section)
conda env create -f environment.yml
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fmfwi

# `conda activate` can report success (and update CONDA_PREFIX) without
# actually putting fmfwi's bin/ ahead of whatever was already on PATH, e.g.
# if PATH was pre-seeded by an IDE/shell outside conda's own bookkeeping.
# Trusting it blindly risks running every pip install below against the
# wrong (possibly unrelated, already-in-use) environment, so verify before
# doing anything else.
ACTIVE_PY_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
EXPECTED_PREFIX="$(conda info --base)/envs/fmfwi"
if [ "$ACTIVE_PY_PREFIX" != "$EXPECTED_PREFIX" ]; then
    echo "ERROR: 'conda activate fmfwi' did not switch the active Python." >&2
    echo "  python in use: $ACTIVE_PY_PREFIX" >&2
    echo "  expected:      $EXPECTED_PREFIX" >&2
    echo "Refusing to continue installing into the wrong environment." >&2
    exit 1
fi
echo 'Created and activated environment:' $(which python)

# detect GPU and install a matching torch build, BEFORE anything that
# transitively depends on torch. Installing torch first keeps pip's resolver
# from backtracking into ancient, no-prebuilt-wheel package versions while
# it searches for a torch to pair them with, and stops those packages from
# silently pulling in a different, CUDA-mismatched torch build of their own
# choosing.
echo 'Detecting GPU...'
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
    echo "GPU detected, driver supports CUDA up to ${DRIVER_CUDA}"

    # Candidate torch wheel builds, newest CUDA requirement first.
    # Update this list as new PyTorch CUDA wheels are published.
    CUDA_BUILDS=("12.4:cu124" "12.1:cu121" "11.8:cu118")
    TORCH_INDEX=""
    DRV_NUM=$(( ${DRIVER_CUDA%%.*} * 10 + ${DRIVER_CUDA##*.} ))
    for build in "${CUDA_BUILDS[@]}"; do
        required="${build%%:*}"
        tag="${build##*:}"
        REQ_NUM=$(( ${required%%.*} * 10 + ${required##*.} ))
        if [ "$DRV_NUM" -ge "$REQ_NUM" ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/${tag}"
            echo "Installing torch for CUDA ${required} (${tag})"
            break
        fi
    done

    if [ -z "$TORCH_INDEX" ]; then
        echo "Driver CUDA ${DRIVER_CUDA} is older than any supported torch build; falling back to CPU-only torch"
        pip install torch --index-url https://download.pytorch.org/whl/cpu
    else
        pip install torch --index-url "$TORCH_INDEX"
    fi
else
    echo 'No GPU detected, installing CPU-only torch'
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# now install devito/devitofwi and this package itself. pytest is required
# here for a real reason, not just for running tests: devito's bundled
# examples.seismic.model module (imported transitively via
# pylops.waveeqprocessing -> devitofwi.waveengine) guards "import pytest"
# with suppress(ImportError) but then uses @pytest.mark.parametrize as a
# *module-level* decorator unconditionally, so the module fails to import at
# all unless pytest happens to be installed.
echo 'Installing devito, devitofwi and fm_fwi...'
pip install \
    pytest \
    devito \
    git+https://github.com/DIG-Kaust/Devito-fwi.git@main
pip install -e .

# check packages work as expected
echo 'Checking devito, devitofwi, torch, fm_fwi and running a command...'
python -c 'import devito; print("devito", devito.__version__);'
python -c 'from devitofwi.waveengine.acoustic import AcousticWave2D; print("devitofwi imported OK");'
python -c 'import torch; print("torch", torch.__version__, "cuda available:", torch.cuda.is_available());'
python -c 'from fm_fwi.model.pretrained import load_pretrained_multidim_wav2vec2; print("fm_fwi imported OK");'

echo 'Done!'
