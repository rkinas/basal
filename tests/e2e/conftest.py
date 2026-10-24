"""End-to-end tests of a basal-ui release: a real basal-ui process (python -m basal.ui.app on a free port with its own
settings directory) and real basal-serve children of the models on disk, driven through basal-ui's HTTP API, its API
server and, for the pages, Chromium via Playwright (basal[e2e]). Opt-in, they take minutes: BASAL_E2E=1. What runs
where, and the variables that pick the models: docs/E2E.md."""
import os
from pathlib import Path

import pytest

from tests.e2e.support import UI, pick, text_ok

HERE = Path(__file__).resolve().parent
ENABLED = os.environ.get("BASAL_E2E") == "1"


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="end-to-end tests load real models: set BASAL_E2E=1 (docs/E2E.md)")
    for item in items:
        if HERE in Path(item.path).parents:
            item.add_marker(pytest.mark.e2e)
            if not ENABLED:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def ui(tmp_path_factory):
    app = UI(tmp_path_factory.mktemp("basal-ui")).start()
    yield app
    left = app.stop()
    assert not left, f"basal-serve children left running after Ctrl+C: {left}"


@pytest.fixture(scope="session")
def hardware(ui):
    return ui.get("/api/hardware")


@pytest.fixture(scope="session")
def text_model(ui):
    """The decision model most tests use: a runnable text model, basal-2.0 first, the smallest."""
    return pick(ui, "BASAL_E2E_MODEL", text_ok, "no runnable text model is downloaded",
                prefer=lambda m: m["family"] != "basal-2.0")


@pytest.fixture(scope="session")
def second_model(ui, text_model):
    return pick(ui, "BASAL_E2E_SECOND_MODEL", lambda m: text_ok(m) and m["id"] != text_model["id"],
                "the multi-model tests need a second runnable text model")


@pytest.fixture(scope="session")
def bf16_model(ui):
    """bf16 PyTorch weights of a text model: the Inspect view and in-process mode comparisons read them."""
    return pick(ui, "BASAL_E2E_BF16_MODEL", lambda m: m["port"] == "bf16" and not m["experimental"]
                and m["family"] in ("basal-2.0", "basal-1.5", "basal-1.0"), "no bf16 text model is downloaded",
                prefer=lambda m: m["family"] != "basal-2.0")
