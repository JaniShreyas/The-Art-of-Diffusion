# Project Rules: ddpm-inpainting

## Learning-First Policy
- This repository is for personal learning and practice of diffusion and flow-matching models; keep agentic coding minimal.
- Do **not** implement or rewrite core model math, backbones, losses, noise schedules, or sampling routines (`src/models/`, `src/models/losses/`, `src/noise_schedules/`, `src/sampling/`) unless explicitly requested.
- For conceptual or debugging questions, explain the math or root cause first and wait for confirmation before editing files.
- Ask before making multi-file edits or adding new abstractions, and do not edit or reformat `notebooks/` or `random_experiments/` unless asked.
- After any edit, briefly explain what changed and where so the user maintains full understanding of the codebase.
- Prioritize clarity and low-resource friendliness (consumer GPU VRAM and runtime).

## Environment & Commands
- Python >= 3.12 managed with `uv` (PyTorch CUDA 12.6 index configured in `pyproject.toml`).
- Always use `uv run --no-sync ...` (or `.venv/Scripts/python.exe`) instead of bare `python` or `pip`, so `uv` does not attempt network syncs in offline/sandboxed environments.
- Lint: `uv run --no-sync ruff check .`
- Train: `uv run --no-sync -m scripts.train` (supports Hydra CLI overrides, e.g., `model=flow_jit_mnist training.epochs=10`).
- Note: `scripts/sample.py` is currently outdated relative to the training loop.

## Repo Map & Key Contracts
- `configs/config.yaml`: Root Hydra config composing `dataset`, `model`, `training`, and `sampling`.
- `configs/model/`, `configs/dataset/`, `configs/training/`, `configs/sampling/`: Modular Hydra YAML configs.
- `src/models/__init__.py`: `MODEL_REGISTRY` and `get_model(config)` factory.
- `src/models/`: Model wrappers and samplers (`ddpm.py`, `edm.py`, `rectified_flow_jit.py`, `autoencoder_kl.py`), `backbones/` (`unet.py`, `unet_attention.py`, `jit.py`, `edm_precond.py`), and `losses/`.
- `src/data/__init__.py` & `src/data/config.py`: `DATASET_REGISTRY` and `DataConfig` dataclass (note: batch size currently lives in the dataset config).
- `src/training/trainer.py`: Main training loop, EMA weights, validation, and FID logging (`Trainer.train()` always logs to MLflow and runs `calculate_fid("final")` at the end).
- `scripts/train.py`: Hydra + MLflow training entrypoint.
- `notebooks/` & `random_experiments/`: Personal exploration and visualizations.
- Model contract: `model(clean_images)` returns `{"loss": tensor}`; `model.sample(num_images, image_size, get_stats, device=...)` handles generation.

## Directories to Ignore
- Do not scan, summarize, or modify `mlruns/`, `data/`, `datasets/`, `output/`, `.hydra/`, `.venv/`, or log files (`train.log`, `debug.log`) unless explicitly requested.
