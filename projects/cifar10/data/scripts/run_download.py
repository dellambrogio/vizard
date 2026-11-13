"""Download the CIFAR-10 data (about 170 MB) and extract it.

The archive and the extracted files go to DATA_DIR (default: .data/cifar10/). Data that is
present is not downloaded again. The server can be slow (the download can take 30 minutes or
more): the log shows the progress, and a download that stopped continues from where it stopped
when you run the script again. The training script also downloads the data on its first run,
so this script is optional.

After the download, the script loads the two splits and logs their sizes.

\b
Examples:
  # Download to .data/cifar10/
  python projects/cifar10/data/scripts/run_download.py
  # Download to another folder
  python projects/cifar10/data/scripts/run_download.py --data-dir /tmp/cifar10
"""

from pathlib import Path

import click
from loguru import logger

from projects.cifar10.data.io import DATA_DIR, download, load_arrays


@click.command(help=__doc__)
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DATA_DIR,
    show_default=True,
    help="Folder for the CIFAR-10 files.",
)
def main(data_dir: Path) -> None:
    download(data_dir)
    for train in (True, False):
        images, labels = load_arrays(train, data_dir)
        split: str = "train" if train else "test"
        logger.info(f"{split}: {len(images)} images {images.shape[1:]}, {len(labels)} labels")
    logger.info(f"CIFAR-10 is in {data_dir}")


if __name__ == "__main__":
    main()
