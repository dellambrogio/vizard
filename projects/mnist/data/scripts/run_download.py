"""Download the MNIST data (about 11 MB).

The four files go to DATA_DIR (default: .data/mnist/). A file that is present is not
downloaded again. A download that stopped continues from where it stopped. The training
script also downloads the data on its first run, so this script is optional.

After the download, the script loads the two splits and logs their sizes.

\b
Examples:
  # Download to .data/mnist/
  python projects/mnist/data/scripts/run_download.py
  # Download to another folder
  python projects/mnist/data/scripts/run_download.py --data-dir /tmp/mnist
"""

from pathlib import Path

import click
from loguru import logger

from projects.mnist.data.io import DATA_DIR, download, load_arrays


@click.command(help=__doc__)
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DATA_DIR,
    show_default=True,
    help="Folder for the MNIST files.",
)
def main(data_dir: Path) -> None:
    download(data_dir)
    for train in (True, False):
        images, labels = load_arrays(train, data_dir)
        split: str = "train" if train else "test"
        logger.info(f"{split}: {len(images)} images {images.shape[1:]}, {len(labels)} labels")
    logger.info(f"MNIST is in {data_dir}")


if __name__ == "__main__":
    main()
