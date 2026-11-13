"""Git state of the code of a job: commit, branch, local changes. Saved in the job folder."""

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger

REPO_ROOT: Path = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class GitInfo:
    """The git state of a repo. dirty: there are staged, unstaged or untracked changes."""

    commit: str
    branch: str
    dirty: bool
    untracked: list[str]


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return result.stdout


def read_git_info(repo: Path = REPO_ROOT) -> tuple[GitInfo, str]:
    """Return the git state of repo and the diff of its changes to HEAD (staged and unstaged).

    Raises OSError if git is not installed, and subprocess.CalledProcessError if repo is not a
    git repo or has no commit.
    """
    commit: str = _git(repo, "rev-parse", "HEAD").strip()
    branch: str = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").strip()
    diff: str = _git(repo, "diff", "HEAD")
    untracked: list[str] = _git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    info = GitInfo(commit=commit, branch=branch, dirty=bool(diff or untracked), untracked=untracked)
    return info, diff


def save_git_info(out_dir: Path, prefix: str = "git", repo: Path = REPO_ROOT) -> GitInfo | None:
    """Write <prefix>.json (GitInfo) and, if there are changes to HEAD, <prefix>.diff to out_dir.

    The diff has the changes to tracked files only; <prefix>.json lists the untracked files.
    If git is not installed or repo is not a git repo, log a warning and return None.
    """
    try:
        info, diff = read_git_info(repo)
    except (OSError, subprocess.CalledProcessError) as e:
        detail: str = e.stderr.strip() if isinstance(e, subprocess.CalledProcessError) else str(e)
        logger.warning(f"no git info for the job: {detail}")
        return None
    (out_dir / f"{prefix}.json").write_text(json.dumps(asdict(info), indent=2) + "\n")
    if diff:
        (out_dir / f"{prefix}.diff").write_text(diff)
    state: str = "with local changes" if info.dirty else "clean"
    logger.info(f"git commit {info.commit[:12]} ({info.branch}), {state}")
    return info
