# ============================================================
# jit-tfg Makefile (rebuttal-exp-hyeongmin branch)
# Adapted from ~/projects/Makefile.template.
# Server-side env setup for the LGD/FreeDoM (Additional TFG) sweep
# and other rebuttal experiments. See ~/projects/CLAUDE.md for the
# common docker workflow.
# ============================================================

# ==================== Project Config ====================

# Match pyproject.toml `requires-python ~=3.11`.
PYTHON := 3.11

# torch>=2.9.1 needs CUDA 12.x; -devel kept conservative in case any
# extra deps need nvcc later (e.g., gsplat). Override on the command line
# if a different image is preferred: make docker DOCKER_IMAGE=...
DOCKER_IMAGE := nvidia/cuda:12.6.3-devel-ubuntu22.04

CONTAINER_NAME := $(notdir $(CURDIR))

# High ports unique to this project (3X range). Adjust on each server if
# already in use by other containers.
#   33060 → TensorBoard (container 6006)
#   33061 → Jupyter      (container 8888)
PORTS := -p 33060:6006 -p 33061:8888

.PHONY: docker docker-init setup run check check-models clean

# ==================== Docker (DO NOT EDIT) ====================

docker:
	@if docker ps --format '{{.Names}}' | grep -qx '$(CONTAINER_NAME)'; then \
		echo "Exec into running container: $(CONTAINER_NAME)"; \
		docker exec -it $(CONTAINER_NAME) bash -l; \
	elif docker ps -a --format '{{.Names}}' | grep -qx '$(CONTAINER_NAME)'; then \
		echo "Starting stopped container: $(CONTAINER_NAME)"; \
		docker start $(CONTAINER_NAME); \
		docker exec -it $(CONTAINER_NAME) bash -l; \
	else \
		echo "Creating new container: $(CONTAINER_NAME)"; \
		docker run --gpus all -d \
			--name $(CONTAINER_NAME) \
			$(PORTS) \
			-v $(CURDIR):/workspace -w /workspace \
			-v $(HOME)/projects/docker-init.sh:/tmp/docker-init.sh:ro \
			$(DOCKER_IMAGE) sleep infinity; \
		docker exec -it $(CONTAINER_NAME) /bin/bash; \
	fi

# ==================== Inside container ====================

# One-time tools install (uv, node, claude code, tmux, git config, etc.)
docker-init:
	bash /tmp/docker-init.sh

# ==================== Project Setup ====================

# Full project setup (run inside container, non-interactive).
# Includes git LFS pull for the FID reference statistics
# (src/jit_tfg/evaluation/generation/fid_stats/**.npz).
setup:
	rm -rf .venv
	uv python install --quiet $(PYTHON)
	uv venv --python $(PYTHON) .venv
	uv sync
	@echo ""
	@echo "=== Ensuring git-lfs is installed (apt-get; container is root) ==="
	@which git-lfs >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y --no-install-recommends git-lfs)
	@echo ""
	@echo "=== Pulling LFS files (FID reference statistics) ==="
	git lfs install
	git lfs pull
	@echo ""
	@echo "=== Setup complete ==="
	@echo "Next:"
	@echo "  1. make check                       - verify env (torch, CUDA, LFS, key imports)"
	@echo "  2. make check-models                - which model checkpoints already on disk"
	@echo "  3. (manual) download missing SiT / JiT-H/16 — see"
	@echo "     experiments/ext_scripts/download_checkpoints.sh"
	@echo ""
	@echo "Run experiments — see docs/rebuttal/new_experiments_hyeongmin.md for full sweep commands:"
	@echo "  EXP-G1 (Additional TFG, bird, Yunsung): finegrained_bird_tfg.py + --guidance_mode {lgd,freedom}"
	@echo "  EXP-G3 (Cars, Hyeongmin):                see docs/rebuttal/exp_g3_stanford_cars.md §6"
	@echo "                                          (needs FID stats precompute + Validity evaluator first)"
	@echo "  EXP-F4 (intra-class variance):           re-generate DPS samples for Fig 6/10 8 conditions, then"
	@echo "                                          uv run --with lpips python experiments/rebuttal/intra_class_diversity.py ..."
	@echo "  EXP-N1 (Δ-Precision):                   already done locally — figure in paper repo"

# ==================== Run (DO NOT EDIT) ====================

run:
	uv run --no-sync python $(SCRIPT) $(ARGS)

# ==================== Check ====================

# Verify Python env, CUDA, key project imports, and LFS files.
check:
	@uv run --no-sync python -c "\
import sys; \
import torch; \
from jit_tfg.evaluation.generation import calculate_fid_from_features, get_finegrained_stats_path; \
from jit_tfg.tfg import TFGConfig; \
from pathlib import Path; \
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}'); \
sz = Path('src/jit_tfg/evaluation/generation/fid_stats/finegrained/child_fid_stats.npz').stat().st_size; \
assert sz > 1_000_000, f'LFS pull missing — file is only {sz} bytes (run: git lfs pull)'; \
assert Path('experiments/finegrained_bird_mapping.json').exists(); \
assert Path('experiments/finegrained_cars_mapping.json').exists(); \
print('=== ALL CHECKS PASSED ===')"

# Verify model checkpoints used by EXP-G1 (LGD/FreeDoM sweep).
# DiT and PixelFlow auto-download from HF; SiT and JiT-H/16 must be
# pre-staged at the paths below (see experiments/ext_scripts/download_checkpoints.sh).
check-models:
	@uv run --no-sync python -c "\
from pathlib import Path; \
import os; \
checks = [ \
    ('SiT-XL/2-256',  'checkpoints/sit_official/SiT-XL-2-256.pt'), \
    ('JiT-H/16',      'checkpoints/jit_official/jit-h-16.pth'), \
    ('JiT-L/16',      'checkpoints/jit_official/jit-l-16.pth'), \
    ('JiT-B/16',      'checkpoints/jit_official/jit-b-16.pth'), \
    ('DiT-XL/2-256',  os.path.expanduser('~/.cache/jit-tfg/dit/DiT-XL-2-256x256.pt')), \
]; \
[print(('   ✓ ' if Path(p).exists() else '   ✗ ') + f'{n:18s} {p}') for n, p in checks]; \
print(); \
print('Auto-download on first use (no action needed if missing): DiT, PixelFlow, two bird classifiers'); \
print('Manual download required if missing: SiT, JiT-H/16 (essential for EXP-G1)')"

# ==================== Clean (DO NOT EDIT) ====================

clean:
	rm -rf .venv
