"""Train a CNN on CIFAR-10 with TorchTNT. Config: Hydra, projects/cifar10/configs/<experiment>.yaml.

Each experiment is one config file. The default is baseline_1.0 (VGG-style CNN). Select an
experiment with --config-name.

The config builds the unit (model, loss, optimizer), the train data, the evaluator and the
exporter: each node with a _target_ key names a class or function, and its other keys are
the arguments.

Each run creates a new job with a unique random id (for example W482915). The job keeps all
artifacts in .jobs/<job id>/: job.json (metadata), job.log (log), config.yaml (resolved config),
git.json and git.diff (git state of the code), model.txt (model summary), checkpoints/ (one per
epoch) and, with trainer.export_onnx=true, export/model.onnx. A resumed run saves its config as
config.resume.yaml. The run also goes to Weights & Biases (project vizard-cifar10, run name =
job id), unless wandb.enabled=false. The CIFAR-10 data (about 170 MB) downloads to
.data/cifar10/ on the first run.

The training hyperparameters are in the hparams node (epochs, batch_size, lr, weight_decay,
label_smoothing, augment). The model values are in the unit.module.cfg node.
Change config values with Hydra overrides (key=value). --cfg job prints the config and exits.

Examples:
  # Train baseline_1.0 (the default) in a new job
  python projects/cifar10/scripts/run_train.py
  # Train 30 epochs with a larger batch and a higher learning rate
  python projects/cifar10/scripts/run_train.py hparams.epochs=30 hparams.batch_size=256 \\
      hparams.lr=2e-3
  # Continue job W482915 until it has 20 epochs in total
  python projects/cifar10/scripts/run_train.py resume=W482915 hparams.epochs=20
  # Start a new job from the weights of job W482915 (fine-tuning)
  python projects/cifar10/scripts/run_train.py init_from=W482915
  # Change the model: values of the config dataclass of the model (ConvNetConfig)
  python projects/cifar10/scripts/run_train.py unit.module.cfg.width=96 \\
      unit.module.cfg.convs_per_stage=3
  # Train without augmentation, with label smoothing
  python projects/cifar10/scripts/run_train.py hparams.augment=false hparams.label_smoothing=0.1
  # Also evaluate on the test set each 100 train steps
  python projects/cifar10/scripts/run_train.py evaluator.every_n_steps=100
  # Train, then export to ONNX and check the export
  python projects/cifar10/scripts/run_train.py trainer.export_onnx=true
  # Train without Weights & Biases
  python projects/cifar10/scripts/run_train.py wandb.enabled=false
  # Quick test run without checkpoints
  python projects/cifar10/scripts/run_train.py hparams.epochs=1 trainer.checkpoint=false \\
      wandb.enabled=false
  # Print the resolved config without training
  python projects/cifar10/scripts/run_train.py hparams.epochs=30 --cfg job
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from projects.cifar10.config import TrainConfig, register
from vizard.training.run import run_training

register()


@hydra.main(version_base="1.3", config_path="../configs", config_name="baseline_1.0")
def main(dict_cfg: DictConfig) -> None:
    cfg = OmegaConf.to_object(dict_cfg)
    assert isinstance(cfg, TrainConfig)
    run_training(dict_cfg, cfg, f"cifar10-{cfg.experiment}")


if __name__ == "__main__":
    main()
