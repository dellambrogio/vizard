"""Train a CartPole policy with PPO and TorchTNT. Config: Hydra, projects/cartpole/configs/.

Each experiment is one config file. The default is baseline_1.0 (actor-critic MLP). Select an
experiment with --config-name.

The config builds the unit (model, PPO loss values, optimizer), the train data (rollouts of the
current policy in Gymnasium environments), the evaluator (greedy episodes) and the exporter:
each node with a _target_ key names a class or function, and its other keys are the arguments.

One epoch is hparams.rollouts_per_epoch rollouts. After each rollout, the unit makes
hparams.ppo_epochs passes over it in minibatches of hparams.batch_size. At the end of each
epoch, the evaluator plays 20 episodes with the greedy policy. A return of 500 is the time limit
of CartPole-v1: the pole stayed up for the full episode.

Each run creates a new job with a unique random id (for example W482915). The job keeps all
artifacts in .jobs/<job id>/: job.json (metadata), job.log (log), config.yaml (resolved config),
git.json and git.diff (git state of the code), model.txt (model summary), checkpoints/ (one per
epoch) and, with trainer.export_onnx=true, export/model.onnx (the policy: input "obs", output
"logits"). The run also goes to Weights & Biases (project vizard-cartpole, run name = job id),
unless wandb.enabled=false.

Change config values with Hydra overrides (key=value). --cfg job prints the config and exits.

Examples:
  # Train baseline_1.0 (the default) in a new job
  python projects/cartpole/scripts/run_train.py
  # Train 30 epochs with a lower learning rate
  python projects/cartpole/scripts/run_train.py hparams.epochs=30 hparams.lr=3e-4
  # Continue job W482915 until it has 30 epochs in total
  python projects/cartpole/scripts/run_train.py resume=W482915 hparams.epochs=30
  # Start a new job from the weights of job W482915 (fine-tuning)
  python projects/cartpole/scripts/run_train.py init_from=W482915
  # Change the model: values of ActorCriticConfig
  python projects/cartpole/scripts/run_train.py unit.module.cfg.hidden=128
  # Train, then export the policy to ONNX and check the export
  python projects/cartpole/scripts/run_train.py trainer.export_onnx=true
  # Quick test run: no W&B
  python projects/cartpole/scripts/run_train.py hparams.epochs=2 wandb.enabled=false
  # Print the resolved config without training
  python projects/cartpole/scripts/run_train.py --cfg job
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from projects.cartpole.config import TrainConfig, register
from vizard.training.run import run_training

register()


@hydra.main(version_base="1.3", config_path="../configs", config_name="baseline_1.0")
def main(dict_cfg: DictConfig) -> None:
    cfg = OmegaConf.to_object(dict_cfg)
    assert isinstance(cfg, TrainConfig)
    run_training(dict_cfg, cfg, f"cartpole-{cfg.experiment}")


if __name__ == "__main__":
    main()
