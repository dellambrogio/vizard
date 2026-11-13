# Project rules

## Environment

- The repo is a uv workspace with four packages, each with its own `pyproject.toml`:
  `ml/vizard/` (package `vizard`), `projects/mnist/` (`vizard-mnist`), `projects/cifar10/`
  (`vizard-cifar10`) and `projects/cartpole/` (`vizard-cartpole`). The root `pyproject.toml`
  lists the members and the tool settings.
- Virtual env: `.venv/` (create or update with `uv sync`). The versions are locked in `uv.lock`.
- Each package lists the dependencies that its own code imports in its `pyproject.toml`
  (`uv add --package <name> <dep>`). A project depends on `vizard`. Dev tools go in the `dev`
  group of the root `pyproject.toml`.
- A new project is also a workspace member: add it to `[tool.uv.workspace]` and to the root
  `dependencies`.
- Checks: `ruff format . && ruff check . && pyrefly check`. Tests: `pytest` from the repo root.

## Code rules

- No `__init__.py` files: packages are implicit namespace packages (PEP 420). Import modules
  directly, for example `from vizard.training.unit import BaseUnit`.
- Logging uses `loguru` (`from loguru import logger`), never `print` or the `logging` module.
- Command-line interfaces use `click`, never `argparse`.
- Exception: training scripts use Hydra for config. YAML configs go in `<module>/configs/`, with a
  typed dataclass schema in `<module>/config.py` registered in Hydra's ConfigStore. Hydra writes
  no output folders; the script saves the resolved config in the job folder.

## Layout

- Tests go in a `tests/` folder beside the code they test, in files named `test_*.py`:
  `ml/vizard/jobs/git.py` → `ml/vizard/jobs/tests/test_git.py`. No repo-root `tests/`.
- Tests use `pytest` only (plain `assert`, fixtures, `parametrize`), never `unittest`.
- Scripts go in a `scripts/` folder beside the code they belong to, in files named `run_*.py`:
  `ml/vizard/export/onnx.py` → `ml/vizard/export/scripts/run_export_onnx.py`. Project-wide
  entrypoints sit at the project level (`projects/mnist/scripts/run_train.py`). Script tests go
  in `scripts/tests/`. No repo-root `scripts/`.

## Git

- Work on `main` only: commit to `main` and push `main`. Do not make branches or PRs.

## Jobs

- Never delete `.jobs/` or anything in it, not even jobs created by your own test runs.
- A job id is `W` plus 6 random digits (`W482915`). It does not depend on the jobs on the
  machine, but two machines can get the same id.
