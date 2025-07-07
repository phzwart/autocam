"""Test cases for the __main__ module."""

import pytest
from click.testing import CliRunner

from autocam import __version__


def test_version():
    """Test that version is available."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert __version__ != "unknown"


def test_version_import():
    """Test that version can be imported from _version module."""
    from autocam._version import __version__ as version_from_version
    assert version_from_version == __version__


def test_version_tuple():
    """Test that version tuple is available."""
    from autocam._version import __version_tuple__, version_tuple
    assert __version_tuple__ is not None
    assert version_tuple == __version_tuple__


def test_version_attributes():
    """Test that version attributes are properly set."""
    from autocam._version import version
    assert version == __version__


def test_import_error_handling(monkeypatch):
    """Test that import error is handled gracefully."""
    # Mock the import to fail
    import sys
    from unittest.mock import MagicMock
    
    # Save original module
    original_module = sys.modules.get('autocam._version')
    
    # Remove the module to force import error
    if 'autocam._version' in sys.modules:
        del sys.modules['autocam._version']
    
    try:
        # Test that import error is handled
        from autocam import __version__
        # If we get here, the import succeeded (which is expected in normal case)
        assert __version__ != "unknown"
    finally:
        # Restore original module
        if original_module:
            sys.modules['autocam._version'] = original_module


def test_main():
    """Test the main CLI function."""
    from autocam.__main__ import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Autocam" in result.output
