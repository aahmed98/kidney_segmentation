"""Entrypoint for the example ML project. Define and run CLI."""
import click

from src.records import test_job_record
from src.train.train import train
from src.inference.infer import infer


@click.group()
def cli() -> None:
    """CLI group to which all specific entrypoints will be added."""
    pass


cli.add_command(train)
cli.add_command(test_job_record)
cli.add_command(infer)


if __name__ == '__main__':
    cli()