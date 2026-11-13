# Vizard

A training framework built on TorchTNT and Hydra, plus the projects that use it.

- `ml/vizard/` is the generic code: training flow, base classes, jobs, tracking.
- `projects/` has one folder per project, which gives the model, data, loss and metrics:
  - [`mnist/`](projects/mnist/README.md): digit classification, the reference project
  - `cifar10/`: image classification, same structure as MNIST
  - [`cartpole/`](projects/cartpole/README.md): reinforcement learning (PPO)

## Setup

Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

## Quick start

```bash
# Train MNIST for 1 epoch, without W&B
python projects/mnist/scripts/run_train.py hparams.epochs=1 wandb.enabled=false

# Export the latest job to ONNX
python ml/vizard/export/scripts/run_export_onnx.py $(ls -t .jobs | head -1)
```

Each run makes a job with a random id (`W482915`) and a folder `.jobs/<job id>/` with the log,
the config, the git state (`git.json`, `git.diff`), the model summary (`model.txt`), the
checkpoints and the export.

## How it works

A project implements these parts; `run_training` (`ml/vizard/training/run.py`) builds them from
the config, sets the seed, makes the job, trains with checkpoints and evaluates.

| Part | Base class | The project implements |
|---|---|---|
| Unit | `BaseUnit` (`training/unit.py`) | the loss (`compute_loss`) |
| Evaluator | `BaseEvaluator` (`evaluation/evaluator.py`) | the metrics |
| Exporter | `BaseExporter` (`export/onnx.py`) | ONNX input/output names, example inputs |
| Train data | `TensorBatchLoader` (in-memory data), `ModelBoundData` (data that needs the model, RL) | the data loading |
| Config | `BaseTrainConfig`, `HParams` (`config.py`) | its hyperparameters |

### Config

Hydra composes the config. A node with `_target_` names a class or function; its other keys are
the arguments:

```yaml
unit:
  module:
    _target_: projects.mnist.models.cnn.Net
    cfg:
      _target_: projects.mnist.models.cnn.NetConfig
      hidden: 128
```

Each experiment is one file (`baseline_1.0.yaml`). Override with `key=value`; print the
resolved config with `--cfg job`. In VS Code, the HydraLance extension makes `_target_` values
clickable.

### Training options

| Option | Config | Default |
|---|---|---|
| Optimizer | `unit.optimizer`: a `torch.optim` class, `vizard.optim.adamw.adamw` or `vizard.optim.muon.muon_adamw` | per project |
| LR scheduler | `unit.lr_scheduler`, e.g. `vizard.optim.schedulers.warmup_cosine`; steps per `hparams.lr_interval` (`epoch` or `step`) | constant LR |
| Gradient clip | `hparams.clip_grad_norm` | off |
| Gradient accumulation | `hparams.grad_accum_steps` | `1` |
| Mixed precision | `trainer.precision`: `bf16` or `fp16` | float32 |
| Weight EMA | `hparams.ema_decay`, `hparams.ema_warmup_steps` | off |
| `torch.compile` | `trainer.compile`, `trainer.compile_mode` | off |
| Seed | `trainer.seed`, `trainer.deterministic` | `0`, off |
| Step checkpoints | `trainer.checkpoint_every_n_steps` (plus one per epoch) | per epoch only |
| ONNX export | `trainer.export_onnx` | off |
| Resume a job | `resume=<job id>` | |
| Start from a job's weights | `init_from=<job id>` | |

A resume mid-epoch restarts that epoch's data from the beginning.

### Weights & Biases

Off in the `vizard` schema; a project turns it on in its config (`wandb` node), with the team
`foobar`. The run name is the job id and the group is the experiment. Needs `wandb login`.

### Multi-GPU (DDP)

```bash
torchrun --nproc-per-node=4 projects/mnist/scripts/run_train.py
# CPU test with 2 processes
CUDA_VISIBLE_DEVICES= torchrun --nproc-per-node=2 projects/mnist/scripts/run_train.py \
  hparams.epochs=1 wandb.enabled=false
```

- `hparams.batch_size` is the global batch; each rank gets `${per_rank:${hparams.batch_size}}`,
  so an experiment has the same batch, LR and steps per epoch on 1 or N GPUs. The batch size must be a multiple of N.
- The train data must be a `ShardableData` (e.g. `TensorBatchLoader`) so each rank gets its own
  part.
- Rank 0 writes the job, W&B and the export. One machine only.

## Add a project

Use `projects/mnist/` as the template:

1. Make `projects/<name>/` with its `pyproject.toml`; add it to `[tool.uv.workspace]` and to
   `dependencies` in the root `pyproject.toml`, then `uv sync`.
2. Add the model, data, loss, and a unit, evaluator and exporter on the `vizard` base classes.
3. Add `config.py`, `configs/common.yaml` and an experiment file.
4. Add `scripts/run_train.py`: one `run_training` call.

## Development

```bash
ruff format . && ruff check . && pyrefly check && pytest
```

Code rules: [`CLAUDE.md`](CLAUDE.md).
