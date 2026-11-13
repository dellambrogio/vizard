"""Jobs: a unique random id like W482915 and an artifact directory under .jobs/<id>/."""

import json
import re
import secrets
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

JOBS_ROOT: Path = Path(__file__).resolve().parents[3] / ".jobs"
ID_PREFIX: str = "W"
ID_DIGITS: int = 6  # random digits after the prefix: 1e6 ids
# Also accepts the ids of a short time with 8 digits or lowercase letters (Ww5gtwa72).
ID_PATTERN: re.Pattern[str] = re.compile(rf"^{ID_PREFIX}(\d{{{ID_DIGITS}}}|[0-9a-z]{{8}})$")
METADATA_FILE: str = "job.json"
LOG_FILE: str = "job.log"


@dataclass(frozen=True)
class Job:
    """A job and its artifact directory."""

    id: str
    name: str
    created: str
    command: list[str]
    root: Path = JOBS_ROOT

    @property
    def dir(self) -> Path:
        """Artifact directory of the job."""
        return self.root / self.id

    def log_to_file(self) -> int:
        """Also write loguru logs to <job dir>/job.log. Return the loguru sink id."""
        return logger.add(self.dir / LOG_FILE)

    def _write_metadata(self) -> None:
        data: dict[str, Any] = asdict(self)
        data.pop("root")
        (self.dir / METADATA_FILE).write_text(json.dumps(data, indent=2) + "\n")


def new_job_id() -> str:
    """Return a random job id: the prefix plus ID_DIGITS random digits, for example W482915."""
    return f"{ID_PREFIX}{secrets.randbelow(10**ID_DIGITS):0{ID_DIGITS}d}"


def create_job(name: str, root: Path = JOBS_ROOT) -> Job:
    """Create a job with a new random id and make its directory.

    The id does not depend on the jobs on this machine. The directory is made with
    exist_ok=False: if the id is in use on this machine, a new id is made. Two machines can
    get the same id (1 in 1e6 for each pair of jobs).
    """
    root.mkdir(parents=True, exist_ok=True)
    while True:
        job_id: str = new_job_id()
        try:
            (root / job_id).mkdir()
        except FileExistsError:
            continue
        break
    job = Job(
        id=job_id,
        name=name,
        created=datetime.now().astimezone().isoformat(timespec="seconds"),
        command=sys.argv,
        root=root,
    )
    job._write_metadata()
    logger.info(f"job {job.id} ({name}): {job.dir}")
    return job


def load_job(job_id: str, root: Path = JOBS_ROOT) -> Job:
    """Load an existing job by id."""
    if not ID_PATTERN.match(job_id):
        raise ValueError(f"invalid job id {job_id!r}, expected {ID_PREFIX} + {ID_DIGITS} digits")
    meta_path: Path = root / job_id / METADATA_FILE
    if not meta_path.exists():
        raise FileNotFoundError(f"job {job_id} not found in {root}")
    data: dict[str, Any] = json.loads(meta_path.read_text())
    return Job(root=root, **data)
