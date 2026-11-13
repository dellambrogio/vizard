# mnist

CNNs that classify MNIST digits. The project trains with TorchTNT and exports the model to ONNX.

Run all commands from the repo root.

## Setup

```bash
uv sync
source .venv/bin/activate
```

Training sends each run to Weights & Biases (team `foobar`, project `vizard-mnist`). Log in
one time:

```bash
wandb login
```

The MNIST data downloads to `.data/mnist/` on the first training run.

## How to train

Each experiment is one config file in `configs/`:

| Experiment | Model | Change from the previous one |
|---|---|---|
| `baseline_1.0` | `Net` in `models/cnn.py`: 2 conv layers, 2 linear layers | The baseline. |
| `candidate_1.1` | `ResNet` in `models/resnet.py`: residual CNN with batch norm, global average pooling, dropout | The model only. |

```bash
# Train baseline_1.0 (the default)
python projects/mnist/scripts/run_train.py

# Train candidate_1.1
python projects/mnist/scripts/run_train.py --config-name candidate_1.1

# Train baseline_1.0 with DDP on 2 GPUs: the same full batch (128), 64 per GPU
torchrun --nproc-per-node=2 projects/mnist/scripts/run_train.py
```

All values that the experiments share are in `configs/common.yaml`. To add an experiment, add
`configs/<name>.yaml` that starts from `common` (see `configs/candidate_1.1.yaml`) and holds the
experiment name, the model and the values that change.

Each run creates a job with a unique random id, for example `W482915`. The log shows the id, and all
files of the job go to `.jobs/<job id>/`:

| File | Content |
|---|---|
| `job.log` | Log of all runs on the job. |
| `config.yaml` | Config of the run, with all values resolved. |
| `git.json`, `git.diff` | Git commit, branch and local changes of the code. A resumed run writes `git.resume.*`. |
| `model.txt` | Model structure and number of parameters. |
| `checkpoints/` | One checkpoint per epoch (and each N train steps with `trainer.checkpoint_every_n_steps`). The last 3 are kept. |
| `wandb/` | Weights & Biases files. |

An evaluation on the full test set runs each 100 train steps and at the end of each epoch.
Each evaluation adds two log lines: the train loss with the accuracy over all images, then the
accuracy of each digit:

```
[mnist] step 100 | train loss 0.4845 | accuracy 0.9554
[mnist]   per digit | 0: 0.996  1: 0.993  2: 0.961  3: 0.947  4: 0.979  5: 0.982  6: 0.980  7: 0.971  8: 0.967  9: 0.966
...
[mnist] epoch 1 | train loss 0.1598 | accuracy 0.9828
[mnist]   per digit | 0: 0.997  1: 0.988  2: 0.993  3: 0.992  4: 0.978  5: 0.970  6: 0.979  7: 0.959  8: 0.989  9: 0.982
```

The loss in a step line is the mean train loss since the previous evaluation. The loss in an
epoch line is the mean train loss of the epoch.

The metrics come from `MnistEvaluator` in `evaluator.py`. The log format comes from
`MnistUnit.log_evaluation` in `unit.py`, which overrides the default line of `BaseUnit`.

### Change the evaluation

```bash
# Evaluate each 50 train steps
python projects/mnist/scripts/run_train.py evaluator.every_n_steps=50

# Evaluate at the end of each epoch only
python projects/mnist/scripts/run_train.py evaluator.every_n_steps=null
```

The `evaluator` node of `configs/common.yaml` holds all of the evaluation: the metrics
(`MnistEvaluator` in `evaluator.py`), the data (`evaluator.loader`, the MNIST test split) and the
interval (`evaluator.every_n_steps`).
For example, to evaluate on the train split:

```bash
python projects/mnist/scripts/run_train.py evaluator.loader.split=train
```

### Change the model

Each model has a config dataclass next to its code. An experiment config names the model and
gives all values of its dataclass in the `unit.module` node:

```yaml
# configs/candidate_1.1.yaml
unit:
  module:
    _target_: projects.mnist.models.resnet.ResNet
    cfg:
      _target_: projects.mnist.models.resnet.ResNetConfig
      width: 32
      blocks_per_stage: 1
      dropout: 0.1
```

| Experiment | Model | Dataclass | Values |
|---|---|---|---|
| `baseline_1.0` | `Net` | `NetConfig` in `models/cnn.py` | `conv1_channels`, `conv2_channels`, `hidden` |
| `candidate_1.1` | `ResNet` | `ResNetConfig` in `models/resnet.py` | `width`, `blocks_per_stage`, `dropout` |

Change the model values in the experiment file, or on the command line with
`unit.module.cfg.<name>=<value>`:

```bash
# baseline_1.0 with a smaller hidden layer
python projects/mnist/scripts/run_train.py unit.module.cfg.hidden=32

# candidate_1.1 with a narrower and deeper model, and no dropout
python projects/mnist/scripts/run_train.py --config-name candidate_1.1 \
    unit.module.cfg.width=16 unit.module.cfg.blocks_per_stage=2 unit.module.cfg.dropout=0.0

# Show the full config of an experiment
python projects/mnist/scripts/run_train.py --config-name candidate_1.1 --cfg job
```

### Change hyperparameters

The training hyperparameters are in the `hparams` node of `configs/common.yaml`. Change them on the
command line with `key=value`:

```bash
# 10 epochs, larger batch, lower learning rate
python projects/mnist/scripts/run_train.py hparams.epochs=10 hparams.batch_size=256 hparams.lr=5e-4

# Label smoothing
python projects/mnist/scripts/run_train.py hparams.label_smoothing=0.1

# Another optimizer
python projects/mnist/scripts/run_train.py unit.optimizer._target_=torch.optim.SGD hparams.lr=0.05

# AdamW with no weight decay on biases and norms
python projects/mnist/scripts/run_train.py unit.optimizer._target_=vizard.optim.adamw.adamw \
  +unit.optimizer.weight_decay=0.01

# Muon on the 2-D weights, AdamW on the output layer (head.3) and the other parameters
python projects/mnist/scripts/run_train.py unit.optimizer._target_=vizard.optim.muon.muon_adamw \
  '+unit.optimizer.adamw_params=[head.3]'

# LR warmup for 1 epoch, then cosine decay until the last epoch
python projects/mnist/scripts/run_train.py hparams.epochs=10 \
  'unit.lr_scheduler={_target_:vizard.optim.schedulers.warmup_cosine,_partial_:true,warmup:1,total:${hparams.epochs}}'

# Gradient clip, gradient accumulation (effective batch 512), bfloat16, another seed
python projects/mnist/scripts/run_train.py hparams.clip_grad_norm=1.0 hparams.grad_accum_steps=4 \
  trainer.precision=bf16 trainer.seed=1

# EMA of the weights for the evaluations and the export, and torch.compile
python projects/mnist/scripts/run_train.py hparams.ema_decay=0.999 trainer.compile=true

# Print the config without training
python projects/mnist/scripts/run_train.py --cfg job
```

### Continue a job

```bash
# The id of the job that changed last
JOB=$(ls -t .jobs | head -1)

# Continue the job until it has 10 epochs in total
python projects/mnist/scripts/run_train.py resume=$JOB hparams.epochs=10

# Start a new job from the weights of the job (fine-tuning): new optimizer, new progress
python projects/mnist/scripts/run_train.py init_from=$JOB hparams.lr=1e-4
```

### Weights & Biases

The run name and id in W&B are the job id, the group is the experiment (`baseline_1.0`,
`candidate_1.1`), and a
resumed job continues the same W&B run.
Runs: https://forge.coreweave.com/wandb/

```bash
# Train without W&B
python projects/mnist/scripts/run_train.py wandb.enabled=false

# Write the W&B run to .jobs/<job id>/wandb/ only, without the W&B server
python projects/mnist/scripts/run_train.py wandb.mode=offline
```

| Metric | When |
|---|---|
| `train/loss`, `train/lr`, `train/grad_norm` (before the clip), `train/steps_per_sec`, `train/data_wait_frac` (part of the time spent waiting for data) | Each 50 train steps (`wandb.log_every_n_steps`; 0 is off). |
| `eval/accuracy`, `eval/accuracy_0` to `eval/accuracy_9` | Each evaluation. |
| `train/loss_epoch`, `train/lr`, `train/epoch_time_sec`, `train/steps_per_sec_epoch`, `train/data_wait_frac_epoch`, `epoch` | Each epoch. |

### Quick trial run

```bash
# 1 epoch, no checkpoints, no W&B
python projects/mnist/scripts/run_train.py hparams.epochs=1 trainer.checkpoint=false wandb.enabled=false
```

## How to export

The export script is generic (`ml/vizard/export/scripts/`): it builds the model and the exporter from
the config of the job. It writes the latest checkpoint of a job to
`.jobs/<job id>/export/model.onnx`, then compares the onnxruntime output with the PyTorch
output.

```bash
# The id of the job that changed last
JOB=$(ls -t .jobs | head -1)

# Export the job and check the result
python ml/vizard/export/scripts/run_export_onnx.py $JOB

# Export without the check
python ml/vizard/export/scripts/run_export_onnx.py $JOB --no-verify

# Export an old job, which has no unit node in its config. The default config is the
# config.yaml of a newer job that used the same model.
python ml/vizard/export/scripts/run_export_onnx.py W000010 --default-config .jobs/$JOB/config.yaml
```

To train and export in one command:

```bash
python projects/mnist/scripts/run_train.py trainer.export_onnx=true
```

The ONNX model has a dynamic batch size:

- Input `image`: float32 `(batch, 1, 28, 28)`, normalized with mean 0.1307 and standard
  deviation 0.3081. The digit is white on a black background.
- Output `logits`: float32 `(batch, 10)`.

## How to evaluate

There is no evaluation script yet. Training reports the test accuracy, in total and per digit,
at each evaluation (log lines and `eval/accuracy*` in W&B); see "Change the evaluation" above.

To try an exported model on images, classify PNG or JPG files with onnxruntime:

```bash
# The id of the job that changed last. The job must have an exported model.
JOB=$(ls -t .jobs | head -1)

# Write the first 20 MNIST test images as PNG files to .data/mnist/images/test/
python projects/mnist/data/scripts/run_extract_images.py --limit 20

# Classify them. Each log line shows the file name, the predicted digit and the confidence.
python projects/mnist/scripts/run_inference.py $JOB .data/mnist/images/test

# Classify your own image. Mostly bright images (dark digit on paper) are inverted.
python projects/mnist/scripts/run_inference.py $JOB path/to/digit.png
```

## More help

Each script shows its full help and examples with `--help`. For the config structure
(`_target_` nodes, schema), see `configs/common.yaml`, `config.py` and `ml/vizard/config.py`.
For the VS Code extension for the Hydra configs, see the [top-level README](../../README.md).
