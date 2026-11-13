"""Train a CNN on MNIST with TorchTNT. Config: Hydra, projects/mnist/configs/<experiment>.yaml.

Each experiment is one config file. The default is baseline_1.0 (small CNN). candidate_1.1 is
a residual CNN. Select an experiment with --config-name.

The config builds the unit (model, loss, optimizer), the train data, the evaluator and the
exporter: each node with a _target_ key names a class or function, and its other keys are
the arguments.

Each run creates a new job with a unique random id (for example W482915). The job keeps all
artifacts in .jobs/<job id>/: job.json (metadata), job.log (log), config.yaml (resolved config),
git.json and git.diff (git state of the code), model.txt (model summary), checkpoints/ (one per
epoch) and, with trainer.export_onnx=true, export/model.onnx. A resumed run saves its config as
config.resume.yaml. The run also goes to Weights & Biases (project vizard-mnist, run name = job
id), unless wandb.enabled=false. The MNIST data downloads to .data/mnist/ on the first run.

The training hyperparameters are in the hparams node (epochs, batch_size, lr,
label_smoothing). The model values are in the unit.module.cfg node.
Change config values with Hydra overrides (key=value). --cfg job prints the config and exits.

Examples:
  # Train baseline_1.0 (the default) in a new job
  python projects/mnist/scripts/run_train.py
  # Train candidate_1.1
  python projects/mnist/scripts/run_train.py --config-name candidate_1.1
  # Train 10 epochs with a larger batch and a lower learning rate
  python projects/mnist/scripts/run_train.py hparams.epochs=10 hparams.batch_size=256 \\
      hparams.lr=5e-4
  # Train with DDP on 2 GPUs: the same full batch (hparams.batch_size), half on each GPU
  torchrun --nproc-per-node=2 projects/mnist/scripts/run_train.py
  # Continue job W482915 until it has 10 epochs in total
  python projects/mnist/scripts/run_train.py resume=W482915 hparams.epochs=10
  # Start a new job from the weights of job W482915 (fine-tuning)
  python projects/mnist/scripts/run_train.py init_from=W482915
  # Change the model: values of the config dataclass of the model (NetConfig, ResNetConfig)
  python projects/mnist/scripts/run_train.py unit.module.cfg.hidden=32
  python projects/mnist/scripts/run_train.py --config-name candidate_1.1 \\
      unit.module.cfg.width=16 unit.module.cfg.blocks_per_stage=2
  # Train with label smoothing
  python projects/mnist/scripts/run_train.py hparams.label_smoothing=0.1
  # Train with another optimizer
  python projects/mnist/scripts/run_train.py unit.optimizer._target_=torch.optim.SGD \\
      hparams.lr=0.05
  # Train, then export to ONNX and check the export
  python projects/mnist/scripts/run_train.py trainer.export_onnx=true
  # Evaluate on the test set each 50 train steps (default: 100), and at the end of each epoch
  python projects/mnist/scripts/run_train.py evaluator.every_n_steps=50
  # Evaluate at the end of each epoch only
  python projects/mnist/scripts/run_train.py evaluator.every_n_steps=null
  # Train without Weights & Biases
  python projects/mnist/scripts/run_train.py wandb.enabled=false
  # Write the W&B run to the job folder only, without the W&B server
  python projects/mnist/scripts/run_train.py wandb.mode=offline
  # Quick test run without checkpoints
  python projects/mnist/scripts/run_train.py hparams.epochs=1 trainer.checkpoint=false \\
      wandb.enabled=false
  # Print the resolved config without training
  python projects/mnist/scripts/run_train.py hparams.epochs=10 --cfg job
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from projects.mnist.config import TrainConfig, register
from vizard.training.run import run_training

register()


@hydra.main(version_base="1.3", config_path="../configs", config_name="baseline_1.0")
def main(dict_cfg: DictConfig) -> None:
    cfg = OmegaConf.to_object(dict_cfg)
    assert isinstance(cfg, TrainConfig)
    run_training(dict_cfg, cfg, f"mnist-{cfg.experiment}")


if __name__ == "__main__":
    main()
