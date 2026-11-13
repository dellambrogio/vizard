"""Classify PNG or JPG digit images with an exported MNIST ONNX model.

MODEL is a job id (the script uses .jobs/<job id>/export/model.onnx) or a path to an .onnx
file. Create the model with ml/vizard/export/scripts/run_export_onnx.py or with
projects/mnist/scripts/run_train.py export_onnx=true.

IMAGES is one image file or a folder with .png/.jpg/.jpeg files (not recursive). Each image is
converted to grayscale and resized to 28x28. MNIST digits are white on black: with
--invert auto (default), mostly bright images (dark digit on light paper) are inverted.

The script logs one line per image: file name, predicted digit and confidence.

\b
Examples:
  # Classify all images in a folder with the model of job W482915
  python projects/mnist/scripts/run_inference.py W482915 .data/mnist/images/test
  # Classify one image with an .onnx file
  python projects/mnist/scripts/run_inference.py path/to/model.onnx path/to/seven.png
  # Images are already white on black: do not invert
  python projects/mnist/scripts/run_inference.py W482915 .data/mnist/images/test --invert no
"""

from pathlib import Path

import click
from loguru import logger

from projects.mnist.runtime.classifier import OnnxClassifier
from projects.mnist.runtime.images import Invert, find_images, load_images
from vizard.jobs.job import load_job


def resolve_model_path(model: str) -> Path:
    """Return the ONNX path for a job id (W482915) or a path to an .onnx file."""
    path = Path(model)
    if path.is_file():
        return path
    return load_job(model).dir / "export" / "model.onnx"


@click.command(help=__doc__)
@click.argument("model")
@click.argument("images", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--invert",
    type=click.Choice([i.value for i in Invert]),
    default=Invert.AUTO.value,
    show_default=True,
    help="Invert colors to get a white digit on black.",
)
@click.option("--batch-size", default=1000, show_default=True, help="Inference batch size.")
def main(model: str, images: Path, invert: str, batch_size: int) -> None:
    try:
        paths: list[Path] = find_images(images)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    if not paths:
        raise click.ClickException(f"no PNG or JPG images in {images}")
    classifier = OnnxClassifier(resolve_model_path(model))
    pred = classifier.predict(load_images(paths, Invert(invert)), batch_size=batch_size)
    for path, label, probs in zip(paths, pred.labels, pred.probs, strict=True):
        logger.info(f"{path.name}: {label} ({probs[label]:.3f})")


if __name__ == "__main__":
    main()
