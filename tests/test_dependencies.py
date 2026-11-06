"""Every third-party module basal imports at module level is a declared dependency of the package: numpy was imported
by the engine, calibration and sets, but came only through transformers and accelerate. The modules of an optional
mode import their extra's package, and are imported only in that mode."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPTIONAL = {"mlx_forward.py": {"mlx", "mlx_lm"},  # --mode mlx (the mlx extra)
            "tpu_kernels.py": {"jax"},  # --mode tpu (the tpu extra)
            "strix.py": {"triton"}}  # the AMD Strix Halo kernels: Triton comes with a ROCm or CUDA torch
DIST = {"mlx_lm": "mlx-lm"}  # import name -> distribution, where they differ


def norm(name):
    return name.lower().replace("_", "-")


def declared():
    """The distributions of pyproject.toml's dependencies (Python 3.10 has no tomllib)."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies = \[(.*?)^\]", text, re.M | re.S)[1]
    block = re.sub(r"#[^\n]*", "", block)
    return {norm(re.split(r"[<>=!~;\[ ]", r, maxsplit=1)[0]) for r in re.findall(r'"([^"]+)"', block)}


def module_imports(path):
    """Third-party top-level packages a module imports at module level (not inside a function or a try)."""
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return {n for n in names if n not in sys.stdlib_module_names and n != "basal"}


def test_every_module_level_import_is_a_declared_dependency():
    have, missing = declared(), {}
    for f in sorted((ROOT / "basal").glob("*.py")):
        need = {norm(DIST.get(n, n)) for n in module_imports(f) - OPTIONAL.get(f.name, set())}
        if need - have:
            missing[f.name] = sorted(need - have)
    assert missing == {}, f"declared: {sorted(have)}"
