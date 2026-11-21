---
name: smoke-test-model
description: >-
  Use this skill when the user asks to verify tensor shapes, check a model's
  forward, backward, or sampling pass, or run a quick sanity check / dry run
  of training without launching a full experiment.
---

# Smoke-Test Model or Training Loop

Keep sanity checks fast, cheap on VRAM, and isolated from real MLflow experiments.

## Option A: Fast Forward / Loss / Sample Check (Preferred)

Run a lightweight one-off script with `uv run --no-sync python -c` that:
1. Composes the target Hydra config using `hydra.initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None)` and `hydra.compose(...)` (avoid `hydra.initialize` inside `python -c`).
2. Instantiates the model with `src.models.get_model(cfg)`.
3. Creates a tiny synthetic batch (`batch_size=2`) matching `cfg.dataset.image_size` and channel count:
   - Check `cfg.model.backbone.in_channels` for UNet/JiT backbones, or `cfg.model.backbone.img_channels` for `EDMPrecond`/SongUNet.
4. Calls `out = model(x)` (which returns `{"loss": tensor}`), verifies `torch.isfinite(out["loss"])`, and optionally calls `out["loss"].backward()` to verify gradient flow.
5. If testing inference/sampling, calls `model.sample(num_images=2, image_size=cfg.dataset.image_size, get_stats=get_stats, device=device)` (with `src.data.get_stats` and `sampling.solver_num_steps=2`) and verifies output shape and `[0, 1]` range.

If `torch` fails to import in `.venv`, stop and tell the user to run `uv sync` rather than modifying `.venv` or triggering network syncs unprompted.

## Option B: End-to-End Training Loop Dry Run

Only run `scripts.train` if the user explicitly asks to test the full dataloader + trainer + MLflow pipeline.

> **Warning:** `Trainer.train()` runs at least 1 full dataset epoch, writes a run and checkpoints to `mlruns/`, and **unconditionally** calls `calculate_fid("final")` after the last epoch (ignoring `calculate_fid_every_n_epochs`).

Always pass these safe Hydra overrides so FID and sampling stay tiny (add `device=cpu` if no GPU is available):

```powershell
uv run --no-sync -m scripts.train experiment_name=smoke_test training.epochs=1 training.validate_every_n_epochs=999 training.calculate_fid_every_n_epochs=999 training.fid_num_images=4 sampling.num_images=4 sampling.solver_num_steps=2
```

## Reporting Rules
- Report shape mismatches, device errors, NaNs, or OOMs clearly with the exact file and line.
- Do **not** rewrite model math to "fix" an issue unless the user asks you to.
