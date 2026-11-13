"""Tests of the git info of a job, on a temporary git repo."""

import json
import subprocess
from pathlib import Path

import pytest

from vizard.jobs.git import save_git_info


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")
    (repo / "a.py").write_text("x = 1\n")
    git(repo, "add", "a.py")
    git(repo, "commit", "-m", "first")
    return repo


def test_clean_repo(repo: Path, tmp_path: Path) -> None:
    info = save_git_info(tmp_path, repo=repo)
    assert info is not None
    assert not info.dirty and info.branch == "main" and len(info.commit) == 40
    assert json.loads((tmp_path / "git.json").read_text())["commit"] == info.commit
    assert not (tmp_path / "git.diff").exists()


def test_changes_go_to_the_diff(repo: Path, tmp_path: Path) -> None:
    (repo / "a.py").write_text("x = 2\n")
    info = save_git_info(tmp_path, prefix="git.resume", repo=repo)
    assert info is not None and info.dirty
    assert "+x = 2" in (tmp_path / "git.resume.diff").read_text()
    assert (tmp_path / "git.resume.json").exists()


def test_staged_changes_go_to_the_diff(repo: Path, tmp_path: Path) -> None:
    (repo / "a.py").write_text("x = 3\n")
    git(repo, "add", "a.py")
    info = save_git_info(tmp_path, repo=repo)
    assert info is not None and info.dirty
    assert "+x = 3" in (tmp_path / "git.diff").read_text()


def test_untracked_files(repo: Path, tmp_path: Path) -> None:
    (repo / "new.py").write_text("y = 1\n")
    info = save_git_info(tmp_path, repo=repo)
    assert info is not None and info.dirty and info.untracked == ["new.py"]
    assert not (tmp_path / "git.diff").exists()  # the diff has tracked files only


def test_not_a_repo(tmp_path: Path) -> None:
    folder = tmp_path / "plain"
    folder.mkdir()
    assert save_git_info(tmp_path, repo=folder) is None
    assert not (tmp_path / "git.json").exists()
