---
name: add-hydra-config
description: >-
  Use this skill when the user asks to add or wire up a new Hydra YAML config
  under configs/ (model, dataset, schedule, training, or sampling) or register
  a new model or dataset variant in MODEL_REGISTRY / DATASET_REGISTRY.
---

# Add Hydra Config & Registry Wiring

Use this procedure to add configuration boilerplate cleanly without altering core model math.

## Steps

1. **Inspect current reference configs first**
   - Check `configs/config.yaml` and a current sibling YAML in the target group:
     - Models: prefer `flow_jit_mnist.yaml`, `ddpm_jit_cifar10.yaml`, or `edm_song_unet_cifar10.yaml` over older templates (`ddpm_mnist.yaml`, `ddpm_cifar10.yaml`) that omit `type` or `defaults: - schedule: ...`.
     - Note that `Trainer.__init__` requires `model.type` (e.g., `"diffusion"`), and `batch_size` currently lives in the dataset config (`src/data/config.py`).

2. **Create the YAML config**
   - Add the new `.yaml` file in the appropriate `configs/<group>/` directory.
   - Keep defaults modest and suitable for consumer GPUs.

3. **Wire up registries (only if adding a new model or dataset type)**
   - **Models:** Add a `create_<model_name>(config)` builder and register it in `MODEL_REGISTRY` inside `src/models/__init__.py`.
   - **Datasets:** Register `(get_dataloaders, get_stats, get_datasets, denormalize)` in `DATASET_REGISTRY` inside `src/data/__init__.py`.
   - Do **not** implement or rewrite underlying model, backbone, or loss math unless explicitly requested.

4. **Verify config resolution (fast, no `torch`/`mlflow` import, no `.hydra/` side effects)**
   - Run Hydra composition directly with `uv run --no-sync`:
     ```powershell
     uv run --no-sync python -c "from pathlib import Path; from hydra import compose, initialize_config_dir; from omegaconf import OmegaConf; initialize_config_dir(config_dir=str(Path('configs').resolve()), version_base=None).__enter__(); print(OmegaConf.to_yaml(compose(config_name='config', overrides=['model=<new_model_config_name>'])))"
     ```
     (Adjust `overrides=[...]` for `dataset=...`, `training=...`, etc.)

5. **Summarize changes clearly**
   - Briefly state which files were created/edited and how Hydra composes the new config into `configs/config.yaml`.
