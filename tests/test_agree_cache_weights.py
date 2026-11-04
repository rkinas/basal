"""basal-agree caches the reference rows: another --gguf / --ollama-model / --ollama-url / --llamacpp-url reads other
weights, so it must not be answered from the rows of an earlier run. The key of a reference that reads none of them
(the default eager-fp32) is the one it was before, so its cached rows stay valid. No network: the model resolution and
the tokenizer are replaced in the test."""
import json

from basal import agree

SINGLE = [([[1, 2], [2, 1]], [[5, 6], [5, 6]]) for _ in range(3)]
MULTI = [([[3, 4], [4, 3]], [[5, 6], [5, 6]])]
LONG = [([[7, 8], [8, 7]], [[5, 6], [5, 6]])]


class Backend:
    def __init__(self, favour):
        self.favour = favour

    def run_shared(self, groups):
        return [[[0.9, 0.1] if self.favour == 0 else [0.1, 0.9] for _ in toks] for toks, _ in groups]


def run(monkeypatch, capsys, cache, *extra, reference="gguf"):
    from transformers import AutoTokenizer
    monkeypatch.setattr("basal.engine.resolve", lambda name: name)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: object(), raising=False)
    monkeypatch.setattr(agree, "build", lambda *a, **k: (SINGLE, MULTI, LONG))
    monkeypatch.setattr(agree, "model_format", lambda md: 1)
    monkeypatch.setattr("basal.server.model_identity", lambda name, md: ("m", "rev"))
    monkeypatch.setattr(agree, "free", lambda be: None)
    monkeypatch.setattr(agree, "backend_factory", lambda spec, md, a, fmt: lambda: Backend(0))
    argv = ["--model", "m", "--reference", reference, "--backends", "fast", "--cache", str(cache),
            "--n-single", "3", "--n-multi", "1", "--n-long", "1", *extra]
    agree.main(argv)
    return json.loads(capsys.readouterr().out.splitlines()[0])["cached"]


def test_reference_rows_are_cached_per_weights(monkeypatch, capsys, tmp_path):
    assert run(monkeypatch, capsys, tmp_path, "--gguf", "A.gguf") is False
    assert run(monkeypatch, capsys, tmp_path, "--gguf", "B.gguf") is False  # other weights: computed again
    assert run(monkeypatch, capsys, tmp_path, "--gguf", "B.gguf") is True  # the same weights: from the cache
    # a gguf reference does not read the llama-server URL
    assert run(monkeypatch, capsys, tmp_path, "--gguf", "B.gguf", "--llamacpp-url", "http://h:1") is True


def test_a_reference_that_reads_none_of_them_keeps_its_cache_when_they_change(monkeypatch, capsys, tmp_path):
    assert run(monkeypatch, capsys, tmp_path, reference="eager-fp32") is False
    other = ("--gguf", "B.gguf", "--ollama-url", "http://h:2")
    assert run(monkeypatch, capsys, tmp_path, *other, reference="eager-fp32") is True
    assert run(monkeypatch, capsys, tmp_path, "--ollama-url", "http://h:2", reference="eager-fp32") is True


def test_an_ollama_reference_depends_on_its_model_and_url_only(monkeypatch, capsys, tmp_path):
    ollama = dict(reference="ollama")
    assert run(monkeypatch, capsys, tmp_path, "--ollama-model", "x", **ollama) is False
    assert run(monkeypatch, capsys, tmp_path, "--ollama-model", "y", **ollama) is False
    assert run(monkeypatch, capsys, tmp_path, "--ollama-model", "y", "--gguf", "A.gguf", **ollama) is True
    assert run(monkeypatch, capsys, tmp_path, "--ollama-model", "y", "--ollama-url", "http://h:2", **ollama) is False


def test_the_key_of_the_default_reference_is_the_one_of_basal_2_5_3():
    sets = {"single": [1]}
    assert agree.cache_key("m", "r", "2.5.3", "eager-fp32", sets) == "107e5bb61e5dbf49c839d9a9"
    assert agree.cache_key("m", "r", "2.5.3", "eager-fp32", sets, {}) == "107e5bb61e5dbf49c839d9a9"
    assert agree.cache_key("m", "r", "2.5.3", "gguf", sets, {"gguf": "A.gguf"}) != \
        agree.cache_key("m", "r", "2.5.3", "gguf", sets, {"gguf": "B.gguf"})
