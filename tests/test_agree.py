"""basal-agree on CPU: its statistics on fixed rows, and the whole tool on a tiny random Llama (eager fp32 as the
native engine, eager bf16 as a backend), with the reference rows cached."""
import json
import re
import zlib

import torch

from basal import agree


def test_compare_and_decisions():
    ref = [[0.9, 0.1], [0.505, 0.495], [0.2, 0.8]]
    got = [[0.89, 0.11], [0.49, 0.51], [0.2, 0.8]]
    r = agree.compare(ref, got)
    assert r["readouts"] == 3 and r["disagreements"] == 1 and r["near_ties"] == 1
    assert abs(r["mean_dp"] - 0.025 / 3) < 1e-12 and abs(r["max_dp"] - 0.015) < 1e-9
    assert [round(x, 6) for x in agree.decisions([[0.7, 0.2, 0.1], [0.3, 0.2, 0.5]])[0]] == [0.6, 0.2, 0.2]


def test_bootstrap_interval():
    units = [1] * 90 + [0] * 10
    lo, hi = agree.bootstrap(units, lambda s: sum(s) / len(s))
    assert lo < 0.9 < hi and hi - lo < 0.15
    assert agree.bootstrap(units, lambda s: sum(s) / len(s)) == [lo, hi]  # seeded
    assert agree.bootstrap([1] * 20, lambda s: sum(s) / len(s)) == [1.0, 1.0]


def test_score_independence_and_accuracy():
    """score: agreement over all sets, decisions of the single questions, accuracy against gold (paired difference),
    and question independence (the backend's request readouts against its own prompts alone)."""
    single = [[[0.8, 0.2], [0.3, 0.7]], [[0.4, 0.6], [0.55, 0.45]]]  # two items, both orders
    multi = [[[0.9, 0.1], [0.2, 0.8]]]
    ref = {"single": single, "multi": multi, "long": []}
    got = {"single": [single[0], [[0.6, 0.4], [0.35, 0.65]]], "multi": multi, "long": []}
    ind = [[0.9, 0.1], [0.6, 0.4]]  # the second prompt alone flips
    row = agree.score(ref, ref, got, ind, gold=[(0, 0), (1, 1)])
    assert row["readouts"] == 6 and row["disagreements"] == 2
    assert row["decision_agreement"] == 0.5
    assert row["reference_accuracy"] == 1.0 and row["accuracy"] == 0.5
    assert row["accuracy_diff_ci"][0] <= -0.5 <= row["accuracy_diff_ci"][1]
    assert row["independence"] == 0.5 and abs(row["independence_max_dp"] - 0.4) < 1e-12
    row["readouts_multi"] = 2
    assert agree.gate(row) == "FAIL"
    assert agree.gate(dict(agreement=1.0, near_ties=0, readouts=4, independence=1.0, independence_ties=0,
                           readouts_multi=2)) == "pass"
    # disagreements on near ties pass: 98 of 100 agree and both flips are near ties
    assert agree.gate(dict(agreement=0.98, near_ties=2, readouts=100)) == "pass"
    assert agree.gate(dict(agreement=0.98, near_ties=0, readouts=100)) == "FAIL"


def test_decisions_follow_the_orders_the_server_answers_with():
    """Prompt-format-2 models are served with one option order (server.default_orders): their decisions are the given
    order's readouts, not the average of both orders."""
    readouts = [[0.7, 0.2, 0.1], [0.3, 0.2, 0.5]]                          # order 1, reversed order
    assert agree.decisions(readouts, orders=1) == [[0.7, 0.2, 0.1]]
    assert [round(x, 6) for x in agree.decisions(readouts, orders=2)[0]] == [0.6, 0.2, 0.2]
    single = [[[0.55, 0.45], [0.8, 0.2]]]                                    # order 1 says A, the reversed order B
    got = [[[0.45, 0.55], [0.8, 0.2]]]                                      # order 1 flips, the average does not
    ref = {"single": single, "multi": [], "long": []}
    two = agree.score(["single"], ref, {"single": got, "multi": [], "long": []}, None, gold=[(0, 0)], orders=2)
    one = agree.score(["single"], ref, {"single": got, "multi": [], "long": []}, None, gold=[(0, 0)], orders=1)
    assert two["decision_agreement"] == 1.0 and one["decision_agreement"] == 0.0
    assert one["reference_accuracy"] == 1.0 and one["accuracy"] == 0.0
    from basal.server import default_orders
    assert default_orders(None, 2) == 1 and default_orders(None, 1) == 2


class Tok:
    """Words and single characters; one-letter codes A-Z get ids 1-26, everything else a hash id in 30-255."""
    pad_token = eos_token = "<pad>"
    pad_token_id = 0
    padding_side = "left"

    def apply_chat_template(self, msgs, **_):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs)

    def _ids(self, text):
        return [ord(t) - 64 if len(t) == 1 and "A" <= t <= "Z" else 30 + zlib.crc32(t.encode()) % 226
                for t in re.findall(r"[A-Za-z]+|.", text, re.S)]

    def __call__(self, text, add_special_tokens=False, **_):
        ids = [t if isinstance(t, list) else self._ids(t) for t in text] if isinstance(text, list) else self._ids(text)
        return type("E", (), {"input_ids": ids})()


def tiny(path):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=16384, tie_word_embeddings=False)
    LlamaForCausalLM(cfg).eval().save_pretrained(path)


def test_agree_on_a_tiny_llama(monkeypatch, tmp_path, capsys):
    """eager fp32 (the native engine) against itself and against eager bf16: identical rows give agreement 1 and
    |dp| 0, bf16 stays close; question independence holds; the second run reads the reference rows from the cache."""
    from transformers import AutoTokenizer

    from basal import bench
    tiny(tmp_path / "m")
    monkeypatch.setenv("BASAL_DEVICE", "cpu")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: Tok())
    built = []
    real = bench.build
    monkeypatch.setattr(bench, "build", lambda mode, md, **kw: built.append(mode) or real(mode, md, **kw))
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps(dict(state="Paczka dotarła uszkodzona.", question="Dział?", options=["Zwroty", "IT"],
                                 gold=0)) + "\n", encoding="utf-8")
    args = ["--model", str(tmp_path / "m"), "--backends", "eager-fp32", "eager", "--n-single", "3", "--n-multi", "1",
            "--n-long", "1", "--cache", str(tmp_path / "cache"), "--questions", str(q), "--out", str(tmp_path / "o.json")]
    assert agree.main(args) == 0
    assert built == ["eager-fp32", "eager-fp32", "eager"]
    out = json.loads((tmp_path / "o.json").read_text())
    same, bf16 = out["rows"]
    assert same["readouts"] == 2 * 4 + 10 + 2 and same["agreement"] == 1.0 and same["max_dp"] < 1e-5
    assert same["independence"] == 1.0 and same["accuracy"] == same["reference_accuracy"]
    assert bf16["agreement"] >= 0.9 and bf16["max_dp"] < 0.1 and bf16["gate"] in ("pass", "FAIL")
    assert set(same["sets"]) == {"single", "multi", "long"}
    assert '"cached": false' in capsys.readouterr().out
    built.clear()
    assert agree.main(args[:3] + ["eager-fp32"] + args[5:]) == 0
    assert built == ["eager-fp32"]  # the reference rows came from the cache
    assert '"cached": true' in capsys.readouterr().out


def test_identity_runs_through_agree(monkeypatch):
    from basal import identity
    seen = []
    monkeypatch.setattr(identity, "main", lambda argv: seen.append(argv) or [{"gate": "pass"}])
    assert agree.main(["--identity", "--model", "m"]) == 0 and seen == [["--model", "m"]]
