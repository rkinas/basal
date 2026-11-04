"""Regression tests for fix/selfcheck-verdicts (Z-1082)."""
import urllib.request

import pytest

from basal import selfcheck as sc


def test_sc4_a_200_that_is_not_json_is_a_clean_exit(monkeypatch):
    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"<html>502</html>"
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: Reply())
    with pytest.raises(SystemExit, match="not JSON"):
        sc.call("http://x", "/v1/models")
