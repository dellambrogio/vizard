"""Export the latest checkpoint of a training job to ONNX. Works for a job of any project.

The model and the exporter are built from the unit and exporter nodes of
.jobs/<job id>/config.yaml. The ONNX file goes to .jobs/<job id>/export/model.onnx. After the
export, the script compares onnxruntime output with PyTorch output.

An old job can have a config with no unit or exporter node. For such a job, give a config
that has them with --default-config: the config.yaml of a newer job that used the same model.

\b
Examples:
  # Export job W482915 and check the result
  python ml/vizard/export/scripts/run_export_onnx.py W482915
  # Export without the onnxruntime check
  python ml/vizard/export/scripts/run_export_onnx.py W482915 --no-verify
  # Export an old job, which has no unit node in its config
  python ml/vizard/export/scripts/run_export_onnx.py W000010 \\
      --default-config .jobs/W482915/config.yaml
"""

from pathlib import Path

import click

from vizard.export.onnx import build_exporter
from vizard.jobs.job import load_job
from vizard.training.run import job_config_node, restore_unit


@click.command(help=__doc__)
@click.argument("job_id")
@click.option("--no-verify", is_flag=True, help="Do not check the ONNX model with onnxruntime.")
@click.option(
    "--default-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="config.yaml of a job, for the unit and exporter nodes, if the job config has none.",
)
def main(job_id: str, no_verify: bool, default_config: Path | None) -> None:
    try:
        job = load_job(job_id)
        job.log_to_file()
        unit = restore_unit(job, default_config)
        exporter = build_exporter(job_config_node(job, "exporter", default_config))
    except KeyError as e:
        raise click.ClickException(f"{e.args[0]} with --default-config") from e
    except (ValueError, FileNotFoundError) as e:
        raise click.ClickException(str(e)) from e
    exporter.run(unit.inference_module(), job.dir / "export" / "model.onnx", verify=not no_verify)


if __name__ == "__main__":
    main()
