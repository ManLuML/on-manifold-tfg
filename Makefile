# ============================================================
# on-manifold-tfg — developer convenience targets
# ============================================================

PYTHON := 3.11

.PHONY: setup lint test check check-models download-stats clean

# Create the virtual environment and install dependencies.
setup:
	uv python install --quiet $(PYTHON)
	uv venv --python $(PYTHON) .venv
	uv sync

# Lint with ruff (same configuration as CI).
lint:
	uv run ruff check src/ experiments/ scripts/ tests/ spiral_test/

# Run the unit test suite (CPU only, no checkpoints required).
test:
	uv run pytest tests/ -q

# Verify the Python environment and key project imports.
check:
	@uv run --no-sync python -c "\
	import torch; \
	from jit_tfg.evaluation.generation import calculate_fid_from_features, get_finegrained_stats_path; \
	from jit_tfg.tfg import TFGConfig; \
	from pathlib import Path; \
	print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}'); \
	assert Path('experiments/finegrained_bird_mapping.json').exists(); \
	stats = Path('src/jit_tfg/evaluation/generation/fid_stats/finegrained/child_fid_stats.npz'); \
	print('FID reference stats: ' + ('present' if stats.exists() else 'missing — run: uv run python scripts/download_fid_stats.py')); \
	print('=== ALL CHECKS PASSED ===')"

# Show which model checkpoints are already on disk.
# DiT auto-downloads on first use; see CHECKPOINTS.md for the rest.
check-models:
	@uv run --no-sync python -c "\
	from pathlib import Path; \
	checks = [ \
	    ('JiT-H/16',      'checkpoints/jit/jit-h-16.pth'), \
	    ('SiT-XL/2-256',  'checkpoints/sit/SiT-XL-2-256.pt'), \
	    ('DiT-XL/2-256',  'checkpoints/dit/DiT-XL-2-256x256.pt'), \
	    ('PixelFlow-XL',  'checkpoints/pixelflow'), \
	]; \
	[print(('   + ' if Path(p).exists() else '   - ') + f'{n:14s} {p}') for n, p in checks]; \
	print(); \
	print('Auto-download on first use: DiT, bird classifiers (HF Hub)'); \
	print('Helper script: uv run python scripts/download_checkpoints.py --all'); \
	print('Manual download: JiT-H/16 — see CHECKPOINTS.md')"

# Fetch the FID reference statistics from the HF Hub dataset.
download-stats:
	uv run python scripts/download_fid_stats.py

clean:
	rm -rf .venv
