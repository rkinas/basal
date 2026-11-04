"""One version everywhere: the package, the Claude Code plugin and the TypeScript client are released together, so a
release that bumps basal/__init__.py and pyproject.toml must bump the others too (they lagged at 2.5.0 in 2.5.3)."""
import json
import re
from pathlib import Path

from basal import __version__

ROOT = Path(__file__).resolve().parent.parent


def test_every_manifest_has_the_package_version():
    pyproject = re.search(r'^version = "([^"]+)"$', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)[1]
    plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))["version"]
    ts = ROOT / "clients" / "typescript"
    package = json.loads((ts / "package.json").read_text(encoding="utf-8"))["version"]
    lock = json.loads((ts / "package-lock.json").read_text(encoding="utf-8"))
    found = {"pyproject.toml": pyproject, ".claude-plugin/plugin.json": plugin, "clients/typescript/package.json": package,
             "clients/typescript/package-lock.json": lock["version"],
             "clients/typescript/package-lock.json (packages)": lock["packages"][""]["version"]}
    assert {k: v for k, v in found.items() if v != __version__} == {}, f"basal {__version__}"
