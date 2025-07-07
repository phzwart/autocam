"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """Autocam."""


if __name__ == "__main__":
    main(prog_name="autocam")  # pragma: no cover
