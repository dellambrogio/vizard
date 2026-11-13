# Vizard

A showcase of a training framework of the kind used at a FAANG company, plus the projects that
use it. It trains, evaluates and exports models with TorchTNT and Hydra, two libraries from Meta.

- `ml/vizard/` is the generic code: the training flow, base classes, jobs, tracking and storage.
- `projects/` has one folder per project. A project gives its model, data, loss and metrics.

The first project is [`projects/mnist/`](projects/mnist/README.md). Its README shows how to
train, export and evaluate. `projects/cifar10/` is image classification with the same structure
as MNIST. [`projects/cartpole/`](projects/cartpole/README.md) is a reinforcement learning
project (PPO): its train data comes from the policy that trains.

## Setup

Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync    # makes .venv/ with the versions in uv.lock
source .venv/bin/activate
```

## Quick start

```bash
# Train the MNIST baseline on this machine only: 1 epoch, no W&B
python projects/mnist/scripts/run_train.py hparams.epochs=1 wandb.enabled=false

# Export the job that changed last to ONNX
JOB=$(ls -t .jobs | head -1)
python ml/vizard/export/scripts/run_export_onnx.py $JOB
```

Each training run makes a job with a random id (for example `W482915`) and a folder
`.jobs/<job id>/` with the log, the config, the git state of the code (`git.json`, `git.diff`),
a model summary (`model.txt`), the checkpoints and the exported model.

## How it works

A project builds a training run from these parts. Each part is a class or function in the
project, and the config says which one to use and with which values.

| Part | Base class | The project implements |
|---|---|---|
| Unit | `BaseUnit` in `ml/vizard/training/unit.py` | the loss (`compute_loss`) and, if necessary, the evaluation log lines |
| Evaluator | `BaseEvaluator` in `ml/vizard/evaluation/evaluator.py` | the metrics; the config gives its data and interval |
| Exporter | `BaseExporter` in `ml/vizard/export/onnx.py` | the ONNX input and output names and example inputs |
| Train data | an iterable of batches: `TensorBatchLoader` in `ml/vizard/data/tensor_loader.py` for data in memory, `ModelBoundData` for data that needs the model (RL) | the data loading |
| Config | `BaseTrainConfig` and `HParams` in `ml/vizard/config.py` | its own hyperparameters |

`run_training` in `ml/vizard/training/run.py` does the remaining work: it builds the parts from the
config, sets the seed, makes the job, saves the config, trains with checkpoints, and evaluates
each N train steps and at the end of each epoch.

The training options that all projects have:

| Option | Config | Default |
|---|---|---|
| Optimizer | `unit.optimizer` (`_partial_: true`): a `torch.optim` class, `vizard.optim.adamw.adamw` (no weight decay on 1-D parameters, an LR for each name pattern) or `vizard.optim.muon.muon_adamw` (Muon on 2-D parameters, AdamW on the others) | set by each project |
| LR scheduler | `unit.lr_scheduler` (`_partial_: true`), for example `vizard.optim.schedulers.warmup_cosine` | `null`: constant LR |
| When the LR scheduler steps | `hparams.lr_interval`: `epoch` or `step` (each optimizer step) | `epoch` |
| Gradient norm clip | `hparams.clip_grad_norm` | `null`: no clip |
| Gradient accumulation | `hparams.grad_accum_steps` | `1` |
| Mixed precision | `trainer.precision`: `bf16` or `fp16` | `null`: float32 |
| Weight EMA, for the evaluations and the export | `hparams.ema_decay` (for example `0.999`), `hparams.ema_warmup_steps` | `null`: no EMA |
| `torch.compile` | `trainer.compile`, `trainer.compile_mode` | `false` |
| Seed | `trainer.seed`, and `trainer.deterministic` to error on operations that are not deterministic | `0`, `false` |
| Checkpoint each N train steps | `trainer.checkpoint_every_n_steps` (also one checkpoint after each epoch) | `null`: after each epoch only |
| ONNX export after training | `trainer.export_onnx` (needs the `exporter` node) | `false` |
| Continue a job | `resume=<job id>` | `null` |
| Start a new job from the weights of a local job (EMA weights if it has EMA) | `init_from=<job id>` | `null` |

A checkpoint does not keep the position in the train data. After a resume from a checkpoint in
the middle of an epoch, the data of that epoch starts again from the start.

### Config

Training uses Hydra. A config node with a `_target_` key names a class or function, and its
other keys are the arguments:

```yaml
unit:
  module:
    _target_: projects.mnist.models.cnn.Net
    cfg:
      _target_: projects.mnist.models.cnn.NetConfig
      hidden: 128
```

Each experiment is one config file (`baseline_1.0.yaml`, `candidate_1.1.yaml`). Change a value
on the command line with `key=value`, and print the full config with `--cfg job`.

### VS Code and the Hydra configs

HydraLance is the best VS Code extension for the Hydra configs. Install it from the Extensions
view (search for "HydraLance"). It needs the Pylance extension and the `.venv` interpreter of
this repo.

With it, `F12` or `Ctrl+click` on a `_target_` value, for example
`projects.mnist.models.cnn.NetConfig`, goes to the Python definition.

### Multi-GPU (DDP)

Start the training script with `torchrun` to train on N GPUs of one machine with DDP:

```bash
torchrun --nproc-per-node=4 projects/mnist/scripts/run_train.py
# Test on CPU with 2 processes (gloo), for example on a machine with one GPU
CUDA_VISIBLE_DEVICES= torchrun --nproc-per-node=2 projects/mnist/scripts/run_train.py \
  hparams.epochs=1 wandb.enabled=false
```

- Each rank trains on its own GPU and its own part of the train data, and DDP averages the
  gradients. `hparams.batch_size` is the full batch of one optimizer step, for any number of
  GPUs: the train data takes `${per_rank:${hparams.batch_size}}`, the batch of one GPU. So an
  experiment gives the same batch, LR and steps per epoch on 1 or N GPUs. The batch size must
  be a multiple of N.
- The train data must be a `ShardableData` (`ml/vizard/data/shardable.py`), for example
  `TensorBatchLoader`, so that each rank gets its own part. A `DataLoader` needs a
  `DistributedSampler`. CartPole rollouts are the same on all ranks (same env seeds), so DDP
  gives CartPole no gain yet.
- Rank 0 makes the job and writes the job files, W&B and the export. Each rank
  evaluates the full eval data; the logged metrics are those of rank 0.
- One machine only: all ranks use the same `.jobs/` folder.

### Tracking

| Feature | Config node | Target |
|---|---|---|
| Weights & Biases | `wandb` | team `TEAMNAME`; the run name is the job id, the group is the experiment |

It is off in the `vizard` schema, and a project turns it on in its config. It needs a login
on the machine: `wandb login`.

## Layout

The repo is a uv workspace. Each package has its own `pyproject.toml` with its dependencies,
and one `uv.lock` at the root pins the versions for all of them.

```
pyproject.toml           workspace root: members, dev tools, ruff/pyrefly/pytest settings
doc/                     documentation (HTML and Markdown)
ml/vizard/               package vizard, imports vizard.*
  pyproject.toml
  config.py              config schema: HParams, TrainerConfig, WandbConfig; per_rank
  data/                  download_file; TensorBatchLoader (fast batches of in-memory tensors);
                         ModelBoundData (data that needs the model); ShardableData (DDP split)
  training/              run.py: run_training; unit.py: BaseUnit; fit.py: fit_unit, restore_latest;
                         distributed.py: DDP helpers; summary.py: model.txt; speed.py: TrainSpeed
  optim/                 adamw.py, muon.py: optimizers; schedulers.py: warmup_cosine
  evaluation/            BaseEvaluator
  export/                onnx.py: BaseExporter; scripts/run_export_onnx.py for a job of any project
  jobs/                  job.py: id and folder in .jobs/; git.py: git.json, git.diff
  tracking/              wandb.py: Weights & Biases
projects/
  mnist/                 package vizard-mnist: MNIST digit classification
  cifar10/               package vizard-cifar10: CIFAR-10 image classification
  cartpole/              package vizard-cartpole: CartPole balance with PPO (RL)
```

## Add a project

1. Make `projects/<name>/` with the model, the data loaders (`TensorBatchLoader` for data in
   memory) and the loss. Copy
   `projects/mnist/pyproject.toml` and change the name, the paths and the dependencies. Add the
   folder to `[tool.uv.workspace]` and the package to `dependencies` in the root
   `pyproject.toml`, then run `uv sync`.
2. Add a unit, an evaluator and an exporter that inherit from the `vizard` base classes.
3. Add `config.py` with the config schema, `configs/common.yaml` and one experiment file.
4. Add `scripts/run_train.py`. Copy the MNIST script: it is one `run_training` call.

Use `projects/mnist/` as the example. [`doc/add_project.html`](doc/add_project.html) shows the
steps, with the code of each file.

## Development

```bash
ruff format . && ruff check . && pyrefly check
```

The code rules are in [`CLAUDE.md`](CLAUDE.md).
