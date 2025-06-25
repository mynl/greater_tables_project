"""
Command line interface: convert files to HTML, text, svg or pdf.
"""

import click
import pandas as pd
from pathlib import Path

from . gtlogging import setup_logging

setup_logging()  # <-- must come before using your package

from .gtconfig import Configurator, write_template
from .gtcore import GT


@click.group()
def cli():
    """Greater Tables CLI tool"""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Write rendered output to file")
@click.option("--format", "-f", type=click.Choice(["html", "text", "latex", "svg", "pdf"]), default="html")
@click.option("--config", type=click.Path(), help="Path to a YAML config file")
def render(input_file, output, format, config):
    """Render a table from a data file."""
    path = Path(input_file)
    ext = path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".feather":
        df = pd.read_feather(path)
    elif ext == ".pkl":
        df = pd.read_pickle(path)
    else:
        raise click.UsageError(f"Unsupported extension: {ext}")

    cfg = Configurator(Path(config) if config else None).get()
    gt = GT(df, config=cfg)

    rendered = (
        gt.render_html() if format == "html"
        else gt.render_text() if format == "text"
        else gt.render_latex()
    )

    if format in ('svg', 'pdf'):
        print('more work to do!!')

    if output:
        Path(output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


@cli.command()
@click.argument("path", type=click.Path(), default="config.yaml")
def write_template(path):
    """Write default config to the given path."""
    write_template(Path(path))
    click.echo(f"Config written to {path}")
