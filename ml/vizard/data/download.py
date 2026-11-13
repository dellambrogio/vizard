"""Download a file over HTTP, with progress in the log and resume of a download that stopped."""

import urllib.request
from pathlib import Path

from loguru import logger

CHUNK_BYTES: int = 1 << 20
PROGRESS_STEP: float = 0.1  # one log line per 10 % of the file


def download_file(url: str, path: Path) -> Path:
    """Download url to path, if path is not present. Return path.

    The data goes to <path>.part first, and the file gets its name only when it is complete,
    so path is never a partial file. If <path>.part is present from a download that stopped,
    the download continues from its end (if the server supports byte ranges).
    """
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part: Path = path.with_name(path.name + ".part")
    start: int = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(request) as response:
        resumed: bool = start > 0 and response.status == 206
        if not resumed:
            start = 0  # a new download, or the server sent the full file again
        total: int = start + int(response.headers.get("Content-Length", 0))
        size_text: str = f"{total / 1e6:.1f} MB" if total else "unknown size"
        from_text: str = f", continuing from {start / 1e6:.1f} MB" if resumed else ""
        logger.info(f"downloading {url} ({size_text}{from_text})")
        done: int = start
        next_log: float = PROGRESS_STEP
        with part.open("ab" if resumed else "wb") as f:
            while chunk := response.read(CHUNK_BYTES):
                f.write(chunk)
                done += len(chunk)
                if total and done / total >= next_log:
                    logger.info(f"  {100 * done / total:.0f} % ({done / 1e6:.1f} MB)")
                    next_log = (done / total) // PROGRESS_STEP * PROGRESS_STEP + PROGRESS_STEP
    if total and done != total:
        raise OSError(f"download of {url} stopped at {done} of {total} bytes; run it again")
    part.rename(path)
    return path
