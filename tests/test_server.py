"""Request handling of the HTTP server without a model: option names reach the prompt, and responses and the model
list conform to the official System One OpenAPI schema (tests/data/systemone_openapi.json)."""
import asyncio
import json
import sys
from pathlib import Path

import jsonschema
import pytest

import basal.bench as bench_module
import basal.server as server_module
from basal.facts import HEADER, facts, inject
from basal.server import Server, certified_table, default_mode, models_payload, named_options, parser, to_items

SPEC = json.loads((Path(__file__).parent / "data/systemone_openapi.json").read_text(encoding="utf-8"))


def schema(name):
    s = json.loads(json.dumps(SPEC["components"]["schemas"][name]).replace("#/components/schemas/", "#/$defs/"))
    s["$defs"] = json.loads(json.dumps(SPEC["components"]["schemas"]).replace("#/components/schemas/", "#/$defs/"))
    return s


class Tok:  # character-level stand-in for the tokenizer
    def apply_chat_template(self, msgs, **_):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        ids = [[ord(c) for c in s] for s in text] if isinstance(text, list) else [ord(c) for c in text]
        return type("E", (), {"input_ids": ids})()


class FakeServer(Server):
    """Server.decide with a stub backend that returns fixed position probabilities for every job."""
    def __init__(self, pos_probs, soam=True, cal=None):
        self.name, self.temps, self.tok, self.letters, self.pos_probs = "basal-test", {}, Tok(), {}, pos_probs
        self.orders, self.soam, self.log, self.cal = 2, soam, None, cal or {}
        self.fmt, self.max_options = 1, 10
        self.tau, self.band = 0.6, (0.2, 0.8)
        self.entries = 0
        self.backend = type("B", (), {"policies": {}})()

    async def _run(self, body):
        self.queue = asyncio.Queue()

        async def worker():
            while True:
                prompts, ids, fut, *_ = await self.queue.get()
                self.entries += 1
                fut.set_result([self.probs_for(p, i) for p, i in zip(prompts, ids)])
        w = asyncio.get_running_loop().create_task(worker())
        await asyncio.sleep(0)
        try:
            return await self.decide(body)
        finally:
            w.cancel()


def _probs_for(self, prompt, ids):
    return self.pos_probs[: len(ids)] if not callable(self.pos_probs) else self.pos_probs(prompt, ids)


FakeServer.probs_for = _probs_for


def test_default_mode_prefers_cuda_over_mlx(monkeypatch):
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server_module.torch.backends.mps, "is_available", lambda: True)
    assert default_mode() == "fast"


def test_default_mode_gemma4_fast_only_where_its_gate_passed(monkeypatch, tmp_path):
    """basal-omni (Gemma 4) without --mode: fast where its fast configuration passed its gate (Hopper with FlashAttention
    3, Blackwell workstation GPUs with flex), eager on other GPUs; every other model fast on any CUDA GPU."""
    omni, llama = tmp_path / "omni", tmp_path / "llama"
    for d, t in ((omni, "gemma4"), (llama, "llama")):
        d.mkdir()
        (d / "config.json").write_text(f'{{"model_type": "{t}"}}', encoding="utf-8")
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: True)
    for kernels in ("fa3", "flex"):
        monkeypatch.setattr(server_module, "gemma_fast_kernels", lambda: kernels)
        assert default_mode(str(omni)) == "fast"
    monkeypatch.setattr(server_module, "gemma_fast_kernels", lambda: None)
    assert default_mode(str(omni)) == "eager"
    assert default_mode(str(llama)) == default_mode() == "fast"


def test_fast_mode_loads_the_eager_fitted_calibration(tmp_path):
    """The graph engine (--mode fast) has no calibration file of its own: CALIBRATION.json, also next to a
    CALIBRATION.fast.json. basal-omni's fast mode passed its automation gate with the eager-fitted file; refits on its
    readouts did not (OMNI.md), so a fast-only file must not be picked up unnoticed."""
    from basal.server import calibration_file
    for name in ("CALIBRATION.json", "CALIBRATION.fast.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert calibration_file(tmp_path, "graph") == tmp_path / "CALIBRATION.json"


@pytest.mark.parametrize("fa3, major, want", [(None, 9, "fa3"), ("no flash_attn_3", 9, None), ("Hopper only", 12, "flex"),
                                              ("Hopper only", 10, None), ("Hopper only", 8, None)])
def test_gemma_fast_kernels_by_gpu_class(monkeypatch, fa3, major, want):
    """basal-omni's fast kernels: fa3 where FlashAttention 3 runs, flex on sm_120 (RTX PRO 6000 class, gated 20 Oct),
    none elsewhere (B200's sm_100 and Ampere have no gate yet)."""
    import basal.engine as engine_module
    monkeypatch.setattr(engine_module, "attention_available", lambda impl: fa3 if impl == "fa3" else None)
    monkeypatch.setattr(engine_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(engine_module.torch.cuda, "get_device_capability", lambda *a: (major, 0))
    assert engine_module.gemma_fast_kernels() == want


def test_gemma_fast_kernels_need_flex(monkeypatch):
    """Without Triton (FlexAttention) neither fast configuration runs: fa3 takes flex in the global layers."""
    import basal.engine as engine_module
    monkeypatch.setattr(engine_module, "attention_available", lambda impl: "no Triton" if impl == "flex" else None)
    assert engine_module.gemma_fast_kernels() is None


def test_gemma_flex_refused_before_loading(tmp_path, monkeypatch):
    import basal.engine as engine_module
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4"}), encoding="utf-8")
    monkeypatch.setattr("basal.engine.AutoTokenizer.from_pretrained", lambda *_: pytest.fail("loaded before refusing"))
    monkeypatch.setattr("basal.engine.set_attention", lambda *_, **__: "fa2")
    monkeypatch.setattr(engine_module, "attention_available", lambda impl: "no Triton" if impl == "flex" else None)
    with pytest.raises(ValueError, match="--gemma-attn flex: no Triton"):
        engine_module.GraphBackend(str(tmp_path), gemma_attn="flex")


def test_default_mode_requires_both_mlx_packages_on_mps(monkeypatch):
    monkeypatch.setattr(server_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(server_module.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    assert default_mode() == "mps"


def test_benchmark_mps_default_avoids_eager_fp32(monkeypatch):
    monkeypatch.setattr(bench_module, "default_mode", lambda model=None: "mlx")
    monkeypatch.setattr(bench_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(bench_module.torch.backends.mps, "is_available", lambda: True)
    assert bench_module.default_modes() == ["mps", "mlx"]


def test_benchmark_cuda_default_preserves_fp32_reference(monkeypatch):
    monkeypatch.setattr(bench_module, "default_mode", lambda model=None: "fast")
    monkeypatch.setattr(bench_module.torch.cuda, "is_available", lambda: True)
    assert bench_module.default_modes() == ["eager-fp32", "fast"]


def test_benchmark_sync_and_memory_follow_backend_device(monkeypatch):
    import math
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(bench_module.torch.mps, "synchronize", lambda: calls.append("mps"))
    monkeypatch.setattr(bench_module.torch.mps, "driver_allocated_memory", lambda: 2**30)
    cpu = SimpleNamespace(dev="cpu")
    mps = SimpleNamespace(dev=bench_module.torch.device("mps"))
    bench_module.timed(lambda: calls.append("work"), 1, cpu)
    assert calls == ["work"]
    assert math.isnan(bench_module.memory_gb(cpu))

    calls.clear()
    bench_module.timed(lambda: calls.append("work"), 1, mps)
    assert calls == ["mps", "work", "mps"]
    assert bench_module.memory_gb(mps) == 1.0


@pytest.mark.parametrize(("mode", "quant"), [
    ("eager", "fp8"), ("fast", "q8"), ("mps", "fp8"), ("mlx", "nvfp4"), ("vllm", "fp8"), ("gguf", "q8"),
    ("mlx", "fp8"), ("mps", "q8"), ("eager", "nvfp4"), ("sglang", "fp8"), ("ollama", "q8"),
])
def test_unsupported_quant_fails_before_model_loading(monkeypatch, mode, quant):
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", mode, "--quant", quant])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    with pytest.raises(SystemExit, match="not supported"):
        server_module.main()


def test_gguf_mode_needs_a_file_before_model_loading(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", "gguf"])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    with pytest.raises(SystemExit, match="--gguf"):
        server_module.main()


def test_missing_calibration_file_fails_before_model_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["basal-serve", "--mode", "eager", "--calibration", str(tmp_path / "typo.json")])
    monkeypatch.setattr(server_module, "resolve", lambda *_, **__: pytest.fail("must reject before loading a model"))
    with pytest.raises(SystemExit, match="no such file"):
        server_module.main()


def test_named_options_show_keys_by_default():
    assert named_options({"returns": "Returns", "it": "IT"}) == (["returns", "it"], ["returns: Returns", "it: IT"])
    assert named_options({"x": None, "y": "why"})[1] == ["x", "y: why"]
    keys, opts = named_options({"approve": {"requires_manager": False}, "reject": {"requires_manager": False}})
    assert opts == ['approve: {"requires_manager": false}', 'reject: {"requires_manager": false}']


def test_named_options_hide_only_on_request():
    assert named_options({"option_1": "Yes", "opcja_2": "Nie"}, "hide")[1] == ["Yes", "Nie"]
    assert named_options({"a": "same", "b": "same"}, "hide")[1] == ["a: same", "b: same"]   # still distinguishable
    with pytest.raises(ValueError):
        named_options({"a": "x", "b": "y"}, "maybe")


def test_structured_choices_give_distinct_prompts():
    q = to_items("Action requested: reject.", {"action": {"type": "choice", "instructions": "Return the action.",
                 "criteria": {"approve": {"requires_manager": False}, "reject": {"requires_manager": False}}}})[0]
    assert len(set(q["options"])) == 2 and q["options"][1].startswith("reject")


def test_force_lang_reaches_yes_no_words_and_branches():
    """The request's "lang" sets the yes/no words and multi branch text too, not only the template (a state that is
    only media, a Polish question without diacritics)."""
    qs = {"cat": {"type": "noul", "instructions": "Czy to kot?"},
          "tags": {"type": "multi", "instructions": "Co widać?", "criteria": {"k": "kot", "p": "pies"}}}
    noul, multi = to_items("", qs, force_lang="pl")
    assert noul["lang"] == multi["lang"] == "pl"
    assert noul["options"] == ["Tak", "Nie"] and noul["branches"] == [("Czy to kot?", ["Tak", "Nie"])]
    assert multi["branches"] == [("Co widać?\nCzy dotyczy: „kot”?", ["Tak", "Nie"]),
                                 ("Co widać?\nCzy dotyczy: „pies”?", ["Tak", "Nie"])]


def test_without_force_lang_the_language_is_detected():
    noul = to_items("", {"cat": {"type": "noul", "instructions": "Czy to kot?"}})[0]   # no diacritics: English
    assert noul["lang"] == "en" and noul["branches"] == [("Czy to kot?", ["Yes", "No"])]
    multi = to_items("", {"t": {"type": "multi", "instructions": "Co widać?", "criteria": {"k": "kot"}}})[0]
    assert multi["lang"] == "pl" and multi["branches"] == [("Co widać?\nCzy dotyczy: „kot”?", ["Tak", "Nie"])]
    qs = {"cat": {"type": "noul", "instructions": "Is it a cat?"}}
    assert to_items("Kot śpi.", qs) == to_items("Kot śpi.", qs, force_lang=None)
    # a Polish state decides every question; an English one leaves it to each question's own text
    mixed = {"a": {"type": "noul", "instructions": "Is it a cat?"}, "b": {"type": "noul", "instructions": "Czy śpi?"}}
    assert [q["lang"] for q in to_items("Kot śpi.", mixed)] == ["pl", "pl"]
    assert [q["lang"] for q in to_items("The cat sleeps.", mixed)] == ["en", "pl"]


def test_request_lang_reaches_the_prompt():
    seen = []
    srv = FakeServer(lambda p, ids: seen.append("".join(map(chr, p))) or [0.5] * len(ids))  # Tok: one id per char
    body = {"state": "", "lang": "pl", "questions": {"cat": {"type": "noul", "instructions": "Czy to kot?"}}}
    asyncio.run(srv._run(body))
    assert seen and all("Tak" in p and "Nie" in p and "Yes" not in p for p in seen)
    with pytest.raises(ValueError, match="lang must be"):
        asyncio.run(srv._run(dict(body, lang="de")))


@pytest.mark.parametrize("q", [
    {"type": "choice", "instructions": "Dept?", "criteria": {"returns": "Returns", "it": "IT"}},
    {"type": "noul", "instructions": "Damaged?"},
    {"type": "score", "instructions": "Urgency?", "criteria": ["low", "mid", "high"]},
])
def test_response_conforms_to_official_schema(q):
    srv = FakeServer([0.7, 0.2, 0.1])
    r = asyncio.run(srv._run({"model": "basal", "state": "Parcel arrived damaged.", "questions": {"q": q}}))
    jsonschema.validate(r, schema("SystemOneResponse"))
    assert isinstance(r["usage"]["input_tokens"], int)


def test_model_list_conforms_to_official_schema():
    jsonschema.validate(models_payload("basal-1.5-4.5B", "fast", {"0.99": None}), schema("ModelMetadataList"))
    m = models_payload("basal-1.5-4.5B", "mlx", {}, {"noul": 1.56})["models"][0]   # basal-calibrate checks it
    assert m["temperature_per_prim"] == {"noul": 1.56}
    assert models_payload("basal-1.5-4.5B", "mlx", {})["models"][0]["temperature_per_prim"] == {}


def test_model_list_carries_certified_thresholds_and_context_policy():
    context = {"limit": 4096, "overflow": "truncate", "truncate_head": 1 / 3}
    payload = models_payload("m", "fast", {}, certified={"0.01": {"choice": 0.9}}, context=context)
    jsonschema.validate(payload, schema("ModelMetadataList"))
    m = payload["models"][0]
    assert m["certified"] == {"0.01": {"choice": 0.9}} and m["context"] == context
    assert "certified" not in models_payload("m", "fast", {}, certified={})["models"][0]
    cal = {"prompt": 2, "thresholds": {"0.01": {"per_type": {"choice": {"confidence": 0.9}, "noul": {}}},
                                       "0.05": {"per_type": {}}}}
    assert certified_table(cal) == {"0.01": {"choice": 0.9}}
    mismatched = models_payload("m", "fast", {}, calibration_warning="belongs to another model",
                                certified={"0.01": {"choice": 0.9}})
    assert "certified" not in mismatched["models"][0]
    assert certified_table({}) == {}


def swapped_prompts_differ(crit_a, crit_b, t="choice"):
    q = lambda c: to_items("Customer selected shipping service B.", {"q": {"type": t, "instructions": "Which code?",
                                                                            "criteria": c}})[0]["options"]
    return q(crit_a) != q(crit_b)


def test_swapping_keys_changes_what_the_model_sees():
    """Regression: whoever owns which description must reach the prompt (short codes, names mentioned in text)."""
    assert swapped_prompts_differ({"approve": "Handled by Alice", "reject": "Handled by Bob"},
                                  {"reject": "Handled by Alice", "approve": "Handled by Bob"})
    assert swapped_prompts_differ({"A": "Dispatch office: Warsaw", "B": "Dispatch office: Krakow"},
                                  {"B": "Dispatch office: Warsaw", "A": "Dispatch office: Krakow"})
    d1, d2 = "Processes approve/reject requests; handled by Alice", "Processes approve/reject requests; handled by Bob"
    assert swapped_prompts_differ({"approve": d1, "reject": d2}, {"reject": d1, "approve": d2})
    assert swapped_prompts_differ({"low": "minor", "high": "severe"}, {"high": "minor", "low": "severe"}, "score")


def test_soam_one_entry_per_request_same_answers():
    body = {"state": "Paczka dotarła uszkodzona, klient żąda zwrotu.", "questions": {
        "dept": {"type": "choice", "instructions": "Dział?", "criteria": {"zwroty": "Zwroty", "it": "IT", "hr": "HR"}},
        "dmg": {"type": "noul", "instructions": "Uszkodzona?"},
        "tags": {"type": "multi", "instructions": "Kategorie?", "criteria": {"d": "Uszkodzenie", "r": "Zwrot"}}}}
    a, b = FakeServer([0.6, 0.3, 0.1]), FakeServer([0.6, 0.3, 0.1], soam=False)
    ra, rb = asyncio.run(a._run(body)), asyncio.run(b._run(body))
    assert a.entries == 1 and b.entries == 4          # one packed row vs one entry per branch
    assert ra["answers"] == rb["answers"] and ra["usage"]["branches"] == 4


def test_act_minimum_expected_cost_and_defer():
    q = {"type": "act", "instructions": "Czy zwrot przysługuje?", "criteria": {"true": "Przysługuje", "false": "Nie"},
         "actions": {"approve": "", "reject": "", "human": ""},
         "costs": {"approve": {"true": 0, "false": 1240}, "reject": {"true": 300, "false": 0},
                   "human": {"true": 15, "false": 15}}}
    # position probabilities: orig order (true, false) and reversed (false, true) -> P(true) = 0.962
    srv = FakeServer(lambda p, ids: [0.962, 0.038] if "A. Przysługuje" in "".join(map(chr, p)) else [0.038, 0.962])
    ans = asyncio.run(srv._run({"state": "Zamówienie 1 240 zł, 6 dni.", "questions": {"r": q}}))["answers"]["r"]
    assert ans["action"] == "human" and abs(ans["expected_costs"]["approve"] - 47.12) < 1e-4
    assert ans["answer"] == "true" and ans["calibration"] == "unvalidated"
    srv = FakeServer(lambda p, ids: [0.995, 0.005] if "A. Przysługuje" in "".join(map(chr, p)) else [0.005, 0.995],
                     cal={"temperature_per_prim": {"noul": 1.0}, "thresholds": {"0.01": {"confidence": 0.999}}})
    ans = asyncio.run(srv._run({"state": "Zamówienie", "questions": {"r": dict(q, max_error=0.01)}}))["answers"]["r"]
    assert ans["refused"] == "approve" and ans["action"] == "human" and ans["calibration"] == "validated"


def test_act_defers_when_no_threshold_is_certified():
    """max_error without a certified threshold for its type and target (no calibration, a null threshold, another
    target, a per-type file without the type) never acts automatically: the defer action is returned, the automatic
    one is named in `refused`, and `deferred` is true."""
    q = {"type": "act", "instructions": "Czy zwrot przysługuje?", "criteria": {"true": "Przysługuje", "false": "Nie"},
         "costs": {"approve": {"true": 0, "false": 1240}, "reject": {"true": 300, "false": 0},
                   "human": {"true": 15, "false": 15}}}
    probs = lambda p, ids: [0.995, 0.005] if "A. Przysługuje" in "".join(map(chr, p)) else [0.005, 0.995]
    cal = {"temperature_per_prim": {"noul": 1.0},
           "thresholds": {"0.01": {"confidence": 0.99}, "0.05": {"confidence": None}}}

    def ask(cal, **kw):
        body = {"state": "Zamówienie", "questions": {"r": dict(q, **kw)}}
        return asyncio.run(FakeServer(probs, cal=cal)._run(body))["answers"]["r"]
    ans = ask(cal, max_error=0.01)                                   # P = 0.995 >= 0.99: automatic, as before
    assert ans["action"] == "approve" and ans["deferred"] is False
    assert ask({})["action"] == "approve"                            # without max_error nothing changes
    for c, target in (({}, 0.01), (cal, 0.05), (cal, 0.02)):
        ans = ask(c, max_error=target)
        assert ans["action"] == "human" and ans["refused"] == "approve" and ans["deferred"] is True
        assert ans["max_error"] == "no certified threshold for this type at this max_error"
    cheap_human = dict(q["costs"], human={"true": 1, "false": 1})    # deferring is already the cheapest action
    ans = ask({}, max_error=0.01, costs=cheap_human)
    assert ans["action"] == "human" and "refused" not in ans and ans["max_error"].startswith("no certified")
    assert ans["deferred"] is True
    # per-type thresholds: the yes/no act uses the noul one only; without it, it defers whatever the pooled one says
    per = {"temperature_per_prim": {"noul": 1.0},
           "thresholds": {"0.01": {"confidence": 0.5, "per_type": {"choice": {"confidence": 0.5}}}}}
    ans = ask(per, max_error=0.01)
    assert ans["action"] == "human" and ans["refused"] == "approve" and ans["deferred"] is True
    per["thresholds"]["0.01"]["per_type"]["noul"] = {"confidence": 0.99}
    assert ask(per, max_error=0.01)["action"] == "approve"
    with pytest.raises(ValueError, match="defer action"):
        ask({}, max_error=0.01, costs={"approve": {"true": 0, "false": 9}, "reject": {"true": 9, "false": 0}})


def test_act_short_form():
    q = {"type": "act", "instructions": "Dept?", "criteria": {"a": "A", "b": "B"}, "costs": {"wrong": 99, "defer": 1}}
    srv = FakeServer([0.9, 0.1])  # the same position probabilities in both orders -> canonical 0.5 / 0.5
    ans = asyncio.run(srv._run({"state": "x", "questions": {"q": q}}))["answers"]["q"]
    assert ans["action"] == "defer" and ans["expected_costs"]["a"] == 49.5


def test_multi_selection():
    labels = {"d": "Uszkodzenie", "r": "Zwrot", "i": "Faktura"}
    p_yes = {"Uszkodzenie": 0.9, "Zwrot": 0.7, "Faktura": 0.1}

    def probs(prompt, ids):
        text = "".join(map(chr, prompt))
        py = next(v for k, v in p_yes.items() if f"„{k}”" in text)
        return [py, 1 - py] if "A. Tak" in text else [1 - py, py]
    srv = FakeServer(probs)
    body = {"state": "Paczka uszkodzona, chcę zwrotu.", "questions": {"t": {"type": "multi", "instructions": "Które?",
                                                                            "criteria": labels, "max": 1}}}
    ans = asyncio.run(srv._run(body))["answers"]["t"]
    assert ans["selected"] == ["d"] and abs(ans["probabilities"]["r"] - 0.7) < 1e-6
    assert abs(ans["set_confidence"] - 0.9 * 0.3 * 0.9) < 1e-6


def test_facts_auto_appends_calendar_facts():
    seen = []

    def probs(prompt, ids):
        seen.append("".join(map(chr, prompt)))
        return [0.6, 0.4]
    srv = FakeServer(probs)
    body = {"state": "Doręczono 20 marca 2027 r., termin 7 dni.", "facts": "auto",
            "questions": {"q": {"type": "noul", "instructions": "Czy w terminie?"}}}
    asyncio.run(srv._run(body))
    assert any("Fakty pomocnicze" in s and "sobota" in s for s in seen)
    with pytest.raises(ValueError):
        asyncio.run(FakeServer([0.5, 0.5])._run(dict(body, facts="yes")))


# money facts: one fixture per pattern, and one negative each
CENNIK = ("Cennik: 174,00 zł za sztukę przy zamówieniu poniżej 20 szt.; 156,60 zł od 20 do 99 szt.; 139,20 zł od 100 szt. "
          "Dostawa kosztuje 39,00 zł, a przy wartości towaru co najmniej 2 000,00 zł jest bezpłatna. Klient zamawia 98 szt.")
VERDICT = ("przekracz", "mieści", "w terminie", "po terminie", "zasadn", "limit", "próg", "przysługuje")


def test_facts_euro_conversion():
    ls = facts("Limit: 2 000 000 euro. Kurs ten (1 października 2026 r.) wynosił 4,3770 zł. Przychód: 9 654 410,27 zł.")
    assert "2 000 000 euro × 4,3770 zł = 8 754 000,00 zł" in ls and "9 654 410,27 zł > 8 754 000,00 zł" in ls
    assert "139 000 euro × 4,2693 zł = 593 432,70 zł" in facts("Próg: 139 000 EUR; 1 EUR = 4,2693 zł.")
    assert facts("Kawa kosztowała 3,50 euro, a obiad 12 euro.") == []                                 # no rate
    assert not any(" euro × " in x for x in facts("Próg 100 000 euro. Kursy: 4,30 zł i 4,50 zł."))   # ambiguous


def test_facts_net_gross_and_comparisons():
    ls = facts("Stawka podatku VAT: 23%. Akceptacja od 1 000,00 zł, 3 000,00 zł i 32 000,00 zł. "
               "Wniosek: laptop — 1 szt. × 26 016,26 zł netto.")
    assert "26 016,26 zł netto × 1,23 = 32 000,00 zł brutto" in ls and "32 000,00 zł = 32 000,00 zł" in ls
    ls = facts("Stawki VAT: 23% i 8%. Próg 3 000,00 zł, 5 000,00 zł, 9 000,00 zł. Zakup 1 szt. × 2 910,00 zł netto.")
    assert not any(" netto × " in x for x in ls)                                                      # two rates
    assert facts("Kwota do zwrotu 78 565,34 zł.") == []                                               # nothing to compare


def test_facts_budget_products_sum():
    ls = facts("Budżet działu wynosi 50 000,00 zł, dotychczas wydano 41 250,00 zł. "
               "Zakup: 3 szt. × 1 250,00 zł netto; 1 usł. × 900,00 zł netto. Stawka VAT 23%.")
    assert {"budżet 50 000,00 zł − wydano 41 250,00 zł = 8 750,00 zł", "3 × 1 250,00 zł = 3 750,00 zł",
            "3 750,00 zł + 900,00 zł = 4 650,00 zł", "5 719,50 zł < 8 750,00 zł"} <= set(ls)
    assert not any(x.startswith("budżet") for x in facts("Wydano decyzję o zwrocie 1 200,00 zł. Opłata 100,00 zł."))


def test_facts_price_list_tier():
    ls = facts(CENNIK)
    assert {"20 ≤ 98 ≤ 99; 98 × 156,60 zł = 15 346,80 zł", "15 346,80 zł + 39,00 zł = 15 385,80 zł",
            "15 346,80 zł > 2 000,00 zł"} <= set(ls)
    assert not any(" ≤ " in x for x in facts(CENNIK.replace("98 szt.", "98 szt., a potem 120 szt.")))  # 2 quantities
    out = inject(CENNIK)
    assert not any(w in out[out.index(HEADER):] for w in VERDICT)                                     # numbers only


# VAT-line regressions
RATES = ["Próg 216 000 euro, przeliczany po kursie 4,31 zł za euro. Wartość 200 000,00 zł netto, VAT 23%.",
         "Kurs ten wynosił 4,31 zł. Próg 216 000 euro. Wartość 200 000,00 zł netto, VAT 23%.",
         "Próg 216 000 euro (4,2586 zł za 1 euro). Wartość 200 000,00 zł netto, VAT 23%.",
         "Próg 216 000 euro; 1 EUR = 4,31 zł. Wartość 200 000,00 zł netto, VAT 23%."]


def test_vat_line_reads_amounts_without_thousands_separator():   # regression: "2026,00 zł" was read as "0,00 zł"
    ls = facts("Faktura na kwotę 2026,00 zł; w tym VAT 23%.")
    assert ls == ["2 026,00 zł × 1,23 = 2 491,98 zł; 2 026,00 zł ÷ 1,23 = 1 647,15 zł"]
    assert facts("Opłata 1500 zł, VAT 8%.")[0].startswith("1 500,00 zł × 1,08 = 1 620,00 zł")


def test_vat_line_skips_exchange_rates():                         # regression: "4,31 zł" got a net/gross line
    for s in RATES:
        ls = facts(s)
        assert not any(x.startswith(("4,31 zł", "4,2586 zł", "4,26 zł")) for x in ls), ls
        assert any(x.startswith("200 000,00 zł × 1,23 = 246 000,00 zł") for x in ls)
    assert facts("Cena 4,31 zł netto, VAT 23%.")[0].startswith("4,31 zł × 1,23")    # a price, not a rate: kept


def test_amounts_with_dot_thousands_separator():                 # regression: "10.000,00 zł" was read as "0,00 zł"
    assert facts("Faktura na 10.000,00 zł, VAT 23%.") == [
        "10 000,00 zł × 1,23 = 12 300,00 zł; 10 000,00 zł ÷ 1,23 = 8 130,08 zł"]
    assert facts("Próg 1.500 zł, zamówienie 1 200 zł.") == ["1 500,00 zł > 1 200,00 zł"]   # was "500,00 zł < 1 200,00 zł"
    assert facts("Kwota 1.234.567,89 PLN brutto, VAT 23%.") == ["1 234 567,89 zł brutto ÷ 1,23 = 1 003 713,73 zł netto"]
    for s in ("Termin 01.05.2026, opłata 1.5 zł, VAT 23%.", "Wersja 2.25 zł, VAT 23%."):   # a date, a decimal: no amount
        assert not any(" zł × " in x or " zł ÷ " in x for x in facts(s)), s
    assert facts("Opłata 1500 zł, VAT 8%.")[0].startswith("1 500,00 zł × 1,08")          # unchanged forms
    assert facts("Faktura na 1 000,00 zł netto, VAT 23%.")[0].startswith("1 000,00 zł × 1,23")


def test_year_of_a_date_is_not_a_duration():                     # regression: "2025 roku" added "+ 1 rok"
    ls = facts("Doręczono 3 marca 2025 roku, termin 7 dni.")
    assert "3 marca 2025 r. + 7 dni (kalendarzowo) = 10 marca 2025 r. (poniedziałek)" in ls
    assert not any(" rok (kalendarzowo)" in x for x in ls), ls
    assert not any(" rok (kalendarzowo)" in x for x in facts("Umowę zawarto 5 maja 2026 r. W 2025 roku nie było zmian."))
    ls = facts("Gwarancja obowiązuje w terminie roku od 3 marca 2025 roku.")                  # a real "roku": kept
    assert "3 marca 2025 r. + 1 rok (kalendarzowo) = 3 marca 2026 r. (wtorek)" in ls
    assert any(" + 1 miesiąc " in x for x in facts("Wpłynęło 3 marca 2025 roku; odpowiedź w terminie miesiąca."))
    for text in ("Doręczono 3 marca 2025  roku, termin 7 dni.", "Doręczono 3 marca 2025\nroku, termin 7 dni.",
                 "Doręczono 3 marca 2025\troku, termin 7 dni.", "Doręczono 2025-03-03 roku, termin 7 dni."):
        assert not any(" rok (kalendarzowo)" in x for x in facts(text)), text


# VAT rate needs VAT context
COURT_FEE = ("Pozew o zapłatę 19 726,00 zł. Opłata: ponad 15 000 do 20 000 zł — 1000 zł; ponad 20 000 zł — 5% wartości "
             "przedmiotu sporu, nie więcej niż 200 000 zł.")



def test_very_long_amounts_do_not_crash_facts():                   # 27+ digits exceeded Decimal's default precision
    for text in ("Faktura na 123.456.789.012.345.678.901.234.567,00 zł, VAT 23%.",
                 "Faktura na 123 456 789 012 345 678 901 234 567 890,00 zł, VAT 23%."):
        assert facts(text)

def test_percentage_is_a_vat_rate_only_with_vat_in_its_sentence():   # regression: "5% wartości sporu" got ×1,05 lines
    assert facts(COURT_FEE) == []
    assert facts("Faktura na 4 877,00 zł. Odsetki wynoszą 13,5% w skali roku. Stopa referencyjna NBP: 5%.") == []
    assert facts("Faktura na 1 000,00 zł netto, VAT 23%.") == ["1 000,00 zł × 1,23 = 1 230,00 zł; 1 000,00 zł ÷ 1,23 = 813,01 zł"]
    assert facts("Faktura: 1 000,00 zł netto. Stawka podatku od towarów i usług: 8%.")[0].startswith("1 000,00 zł × 1,08")


def test_max_len_flag_for_vllm_and_sglang():
    assert parser().parse_args([]).max_len == 4096
    assert parser().parse_args(["--mode", "sglang", "--max-len", "32768"]).max_len == 32768


def test_viz_segments_rebuild_the_prompt():
    from basal.prompt import render
    from basal.viz import segments
    opts = ["Reklamacje kart", "Kredyty"]
    p = render(Tok(), "Klient: błąd hasła.", "Który dział?", opts)
    seg = segments(p, "Klient: błąd hasła.", "Który dział?", opts)
    assert "".join(s["text"] for s in seg) == p
    assert [s["kind"] for s in seg if s["kind"] != "template"] == ["state", "question", "option", "option", "prefill"]


def test_viz_segments_format_2_codes():
    """Prompt format 2 (basal-2.0): option codes past Z, and the segments still rebuild the prompt."""
    from basal.prompt import render
    from basal.viz import segments
    opts = [f"Dział {i}" for i in range(30)]
    p = render(Tok(), "Klient: błąd hasła.", "Który dział?", opts, fmt=2)
    seg = segments(p, "Klient: błąd hasła.", "Który dział?", opts, fmt=2)
    assert "".join(s["text"] for s in seg) == p
    kinds = [s["kind"] for s in seg if s["kind"] != "template"]
    assert kinds == ["state", "question"] + ["option"] * 30 + ["prefill"]
    assert [s["text"] for s in seg if s["kind"] == "option"][27] == "AB. Dział 27"


def test_viz_packing_and_served_comparison():
    from basal.viz import compare_served, packing
    assert packing([[1, 2, 3, 4], [1, 2, 5, 6, 7]]) == {"prefix": 2, "tails": [2, 3], "separate": 9}
    assert packing([[1, 2, 3]]) == {"prefix": 0, "tails": [3], "separate": 3}  # one option order: nothing to share
    res = {"questions": [
        {"qname": "dept", "label_key": None, "keys": ["it", "hr"], "calibrated": [0.75, 0.25]},
        {"qname": "tags", "label_key": "d", "keys": ["true", "false"], "calibrated": [0.6, 0.4]}]}
    served = {"answers": {"dept": {"probabilities": {"it": 0.7, "hr": 0.3}},
                          "tags": {"probabilities": {"d": 0.625, "z": 0.1}}}}
    out = compare_served(res, served)
    assert abs(out["questions"][0]["served_max_diff"] - 0.05) < 1e-9
    assert abs(out["questions"][1]["served_max_diff"] - 0.025) < 1e-9 and out["served"] is served["answers"]


def test_out_of_memory_frees_the_failed_forward():
    """An out-of-memory error's frames are cleared before the requests get it: the failed forward's tensors are freed at
    once, not at a later garbage collection (until then the next requests ran out of memory too; ultra eager, 20 Oct)."""
    import weakref

    import torch
    from basal.server import _fail
    held = []

    def forward():
        t = torch.zeros(4)
        held.append(weakref.ref(t))
        raise torch.OutOfMemoryError("CUDA out of memory")

    async def main():
        fut = asyncio.get_running_loop().create_future()
        try:
            forward()
        except Exception as e:  # noqa: BLE001
            _fail([fut], e)
        assert held[0]() is None
        with pytest.raises(torch.OutOfMemoryError):
            await fut
    asyncio.run(main())
