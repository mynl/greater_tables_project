"""Inject latest version from git describe into toml."""
# inject_version.py

import setuptools_scm
import tomlkit
from pathlib import Path

def main():
    root = Path(__file__).parent
    pyproject_path = root / "pyproject.toml"
    version_file = root / "greater_tables" / "_version.py"

    # Get version from Git
    version = setuptools_scm.get_version(root=str(root))

    # Update pyproject.toml
    pyproject = tomlkit.parse(pyproject_path.read_text())
    pyproject["project"]["version"] = version
    pyproject_path.write_text(tomlkit.dumps(pyproject))
    print(f"✅ Injected version: {version}")

    # Write to _version.py
    version_file.write_text(f'__version__ = "{version}"\n')
    print(f"✅ Wrote version to {version_file}")

if __name__ == "__main__":
    main()
