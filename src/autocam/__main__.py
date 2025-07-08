"""Command-line interface."""

import click

from .conference import create_conference
from .config import load_conference_config
from .config import validate_yaml_schema


@click.group()
@click.version_option()
def main() -> None:
    """Autocam - Representation Learning Conference System."""
    pass


@main.command()
@click.option(
    "--output", "-o", default="example_conference.yaml", help="Output YAML file path"
)
@click.option(
    "--schema",
    "-s",
    default="schemas/example_conference.yaml",
    help="Schema file to use",
)
def create_example(output: str, schema: str) -> None:
    """Create an example YAML configuration file from a schema."""
    import shutil
    from pathlib import Path

    schema_path = Path(schema)
    if not schema_path.exists():
        click.echo(f"❌ Schema file not found: {schema}")
        click.echo("Available schemas:")
        schemas_dir = Path("schemas")
        if schemas_dir.exists():
            for schema_file in schemas_dir.glob("*.yaml"):
                click.echo(f"  - {schema_file}")
        else:
            click.echo("  No schemas directory found")
        raise click.Abort()

    shutil.copy(schema_path, output)
    click.echo(f"✅ Created example configuration from {schema}: {output}")


@main.command()
@click.argument("yaml_file", type=click.Path(exists=True))
def validate(yaml_file: str) -> None:
    """Validate a YAML configuration file."""
    if validate_yaml_schema(yaml_file):
        click.echo("✅ Configuration is valid!")
    else:
        click.echo("❌ Configuration is invalid!")
        raise click.Abort()


@main.command()
@click.argument("yaml_file", type=click.Path(exists=True))
def run(yaml_file: str) -> None:
    """Run a conference directly from YAML file."""
    try:
        conference = create_conference(yaml_file)
        click.echo(f"✅ Loaded conference: {conference.name}")
        click.echo(f"Sessions: {conference.list_sessions()}")
        click.echo(f"Participants: {conference.list_participants()}")
        click.echo("\n💡 To use this conference:")
        click.echo("  from autocam.conference import create_conference")
        click.echo(f"  conference = create_conference({yaml_file!r})")
        click.echo("  # Register your models with conference.register_participant()")
        click.echo("  # Run sessions with conference.run_session()")
    except Exception as e:
        click.echo(f"❌ Error loading conference: {e}")
        raise click.Abort() from e


def print_session_info(session, indent=0):
    """Print detailed information about a session."""
    prefix = "  " * indent
    click.echo(f"{prefix}- {session.name}")
    if session.description:
        click.echo(f"{prefix}  Description: {session.description}")
    if session.training_epochs:
        click.echo(f"{prefix}  Training epochs: {session.training_epochs}")
    if session.working_groups:
        for wg in session.working_groups:
            click.echo(f"{prefix}  Working Group: {wg.name}")
            if wg.description:
                click.echo(f"{prefix}    Description: {wg.description}")
            click.echo(f"{prefix}    Participants: {', '.join(wg.participants)}")
            click.echo(f"{prefix}    Training Mandate: {wg.training_mandate}")
            if wg.student_participants:
                click.echo(
                    f"{prefix}    Students: {', '.join(wg.student_participants)}"
                )
            if wg.target_participants:
                click.echo(f"{prefix}    Targets: {', '.join(wg.target_participants)}")
            if wg.fixed_target:
                click.echo(f"{prefix}    Fixed Target: {wg.fixed_target}")
            if wg.random_mean_count:
                click.echo(f"{prefix}    Random Mean Count: {wg.random_mean_count}")
            if wg.barycentric_min_models and wg.barycentric_max_models:
                click.echo(
                    f"{prefix}    Barycentric Models: "
                    f"{wg.barycentric_min_models}-{wg.barycentric_max_models}"
                )
            if wg.training_epochs:
                click.echo(f"{prefix}    Training epochs: {wg.training_epochs}")
    if session.subsessions:
        click.echo(f"{prefix}  Subsessions:")
        for subsession in session.subsessions:
            print_session_info(subsession, indent + 2)


@main.command()
@click.argument("yaml_file", type=click.Path(exists=True))
def info(yaml_file: str) -> None:
    """Show information about a conference configuration."""
    try:
        config = load_conference_config(yaml_file)
        click.echo(f"Conference: {config.name}")
        if config.description:
            click.echo(f"Description: {config.description}")
        click.echo(f"\nParticipants ({len(config.participants)}):")
        for participant in config.participants:
            click.echo(
                f"  - {participant.name} ({participant.model_tag}, "
                f"{participant.dimension})"
            )
        click.echo("\nParallel Sessions:")
        for session in config.parallel_sessions:
            print_session_info(session, indent=1)
    except Exception as e:
        click.echo(f"❌ Error reading configuration: {e}")
        raise click.Abort() from e


@main.command()
def list_schemas() -> None:
    """List available schema files."""
    from pathlib import Path

    schemas_dir = Path("schemas")
    if not schemas_dir.exists():
        click.echo("❌ No schemas directory found")
        raise click.Abort()
    schema_files = list(schemas_dir.glob("*.yaml"))
    if not schema_files:
        click.echo("❌ No schema files found in schemas/")
        raise click.Abort()
    click.echo("📋 Available schema files:")
    for schema_file in sorted(schema_files):
        click.echo(f"  - {schema_file}")
    click.echo("\n💡 Usage:")
    click.echo("  autocam create-example -s schemas/minimal_conference.yaml")
    click.echo("  autocam run schemas/example_conference.yaml")


if __name__ == "__main__":
    main(prog_name="autocam")  # pragma: no cover
