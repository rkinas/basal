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


def test_sc1_a_near_tie_is_forgiven_only_when_the_server_picks_one_of_the_tied_pair():
    assert sc.compare({"a": .40, "b": .39, "c": .21}, {"a": .10, "b": .20, "c": .70})[1] is False
    assert sc.compare({"a": .40, "b": .39, "c": .21}, {"a": .38, "b": .40, "c": .22})[1] is True
    assert sc.compare({"a": .6, "b": .3, "c": .1}, {"a": .1, "b": .3, "c": .6})[1] is False
