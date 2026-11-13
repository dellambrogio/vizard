"""Write MNIST images as PNG files, for viewing or as input for run_inference.py.

The images go to OUTPUT/<split>/ (default OUTPUT: .data/mnist/images/), one flat folder per
split. File names are <index>_label<digit>.png, for example 00000_label7.png. Images are 28x28
grayscale, white digit on black, as in MNIST. The MNIST data downloads on the first run.

\b
Examples:
  # Write all 10000 test images
  python projects/mnist/data/scripts/run_extract_images.py
  # Write the first 100 training images
  python projects/mnist/data/scripts/run_extract_images.py --split train --limit 100
  # Write both splits to another folder
  python projects/mnist/data/scripts/run_extract_images.py --split all --output /tmp/mnist_png
"""

from pathlib import Path

import click
from loguru import logger
from PIL import Image

from projects.mnist.data.io import DATA_DIR, load_arrays

SPLITS: dict[str, list[bool]] = {"test": [False], "train": [True], "all": [True, False]}


@click.command(help=__doc__)
@click.option(
    "--split",
    type=click.Choice(list(SPLITS)),
    default="test",
    show_default=True,
    help="Which split to write.",
)
@click.option("--limit", type=int, default=None, help="Write only the first N images per split.")
@click.option(
    "--output",
    type=click.Path(file_okay=False, path_type=Path),
    default=DATA_DIR / "images",
    show_default=True,
    help="Output folder.",
)
def main(split: str, limit: int | None, output: Path) -> None:
    for train in SPLITS[split]:
        images, labels = load_arrays(train)
        if limit is not None:
            images, labels = images[:limit], labels[:limit]
        out_dir: Path = output / ("train" if train else "test")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (image, label) in enumerate(zip(images, labels, strict=True)):
            Image.fromarray(image).save(out_dir / f"{i:05d}_label{label}.png")
        logger.info(f"wrote {len(images)} images to {out_dir}")


if __name__ == "__main__":
    main()
