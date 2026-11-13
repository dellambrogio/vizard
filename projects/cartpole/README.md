# CartPole (reinforcement learning)

Train a policy for Gymnasium `CartPole-v1` with PPO: push the cart left or right so the pole
stays up. An episode stops when the pole falls or after 500 steps; the return is the number of
steps. Run all commands from the repo root.

## Train

```bash
# Trial run on this machine only: no W&B (about 15 s)
python projects/cartpole/scripts/run_train.py wandb.enabled=false

# Train and export the policy to ONNX
python projects/cartpole/scripts/run_train.py trainer.export_onnx=true
```

`--help` shows more examples. The baseline gets a return of 500 in all 20 evaluation episodes
after about 5 to 10 epochs.

## How RL fits vizard

vizard trains on an iterable of batches. In RL, the data comes from the policy that trains, so
the train data is `RolloutLoader` (`data/rollouts.py`), not a fixed dataset:

- `run_training` gives it the model (`bind`, see `ModelBoundData` in `ml/vizard/data/model_bound.py`).
- Each epoch, it collects `hparams.rollouts_per_epoch` rollouts with the current policy in
  `hparams.num_envs` environments, and computes the GAE advantages.
- After each rollout, it yields `hparams.ppo_epochs` passes over the rollout in shuffled
  minibatches. One train step is one minibatch update with the PPO loss (`loss/ppo.py`).

The evaluator (`evaluator.py`) plays one episode per seed with the greedy policy and reports
`mean_return`, `min_return` and `success_rate` (return 500). Its data is a list of seeds
(`data/episodes.py`).

## Files

| File | Content |
|---|---|
| `models/actor_critic.py` | `ActorCritic`: policy MLP (`forward` gives the logits) and value MLP |
| `data/rollouts.py` | `RolloutLoader`, `PpoBatch`, `compute_gae` |
| `data/episodes.py` | evaluation seeds and `play_episodes` |
| `loss/ppo.py` | clipped PPO loss, value loss and entropy bonus |
| `unit.py`, `evaluator.py` | `CartPoleUnit`, `CartPoleEvaluator` |
| `export/onnx.py` | ONNX export of the policy: input `obs` (batch, 4), output `logits` (batch, 2) |
| `config.py`, `configs/` | config schema (`PpoHParams`) and experiments |
