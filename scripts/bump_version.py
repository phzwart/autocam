#!/usr/bin/env python3
"""Version bumping script for autocam."""

import re
import sys
from pathlib import Path


def bump_version(version_type):
    """Bump version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.exists():
        print("Error: pyproject.toml not found")
        sys.exit(1)

    # Read current version
    content = pyproject_path.read_text()
    version_match = re.search(r'version = "([^"]+)"', content)

    if not version_match:
        print("Error: Could not find version in pyproject.toml")
        sys.exit(1)

    current_version = version_match.group(1)
    print(f"Current version: {current_version}")

    # Parse version components
    if "." not in current_version:
        print("Error: Invalid version format")
        sys.exit(1)

    parts = current_version.split(".")
    if len(parts) < 2:
        print("Error: Invalid version format")
        sys.exit(1)

    major = int(parts[0])
    minor = int(parts[1])
    patch = int(parts[2]) if len(parts) > 2 else 0

    # Bump version based on type
    if version_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif version_type == "minor":
        minor += 1
        patch = 0
    elif version_type == "patch":
        patch += 1
    else:
        print("Error: Invalid version type. Use 'major', 'minor', or 'patch'")
        sys.exit(1)

    new_version = f"{major}.{minor}.{patch}"
    print(f"New version: {new_version}")

    # Update pyproject.toml
    new_content = re.sub(r'version = "[^"]+"', f"version = {new_version!r}", content)
    pyproject_path.write_text(new_content)

    print(f"Updated pyproject.toml to version {new_version}")
    return new_version


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py [major|minor|patch]")
        sys.exit(1)

    version_type = sys.argv[1]
    new_version = bump_version(version_type)

    print("\nTo create a release:")
    print("1. git add pyproject.toml")
    print(f"2. git commit -m 'Bump version to {new_version}'")
    print(f"3. git tag v{new_version}")
    print("4. git push --tags")
