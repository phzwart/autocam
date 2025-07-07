"""Test cases for the __main__ module."""

from click.testing import CliRunner

from autocam import __version__


def test_version() -> None:
    """Test that version is available."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert __version__ != "unknown"


def test_version_import() -> None:
    """Test that version can be imported from _version module."""
    from autocam._version import __version__ as version_from_version

    assert version_from_version == __version__


def test_version_tuple() -> None:
    """Test that version tuple is available."""
    from autocam._version import __version_tuple__
    from autocam._version import version_tuple

    assert __version_tuple__ is not None
    assert version_tuple == __version_tuple__


def test_version_attributes() -> None:
    """Test that version attributes are properly set."""
    from autocam._version import version

    assert version == __version__


def test_main() -> None:
    """Test the main CLI function."""
    from autocam.__main__ import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Autocam" in result.output
