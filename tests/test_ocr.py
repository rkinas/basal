"""External OCR + generated criteria: contract, inference routing and failure tests."""
import asyncio
import base64
import copy
import io
import json

import httpx
import pytest

from basal import ocr
from basal.prompt import RequestError
from basal.server import Server, error_body, parser, to_items


class Reader:
    def __init__(self, lines):
        self.lines = lines
        self.calls = []

    async def read(self, data, index):
        self.calls.append((data, index))
        return self.lines


class Fake(Server):
    def __init__(self, lines, *, other=False, max_options=10, configured=True):
        self.ocr_client = Reader(lines) if configured else None
        self.max_options = max_options
        self.name = "basal-test"
        self.called = 0
        self.last = None
        self.pick_other = other
        self.in_flight = 0

    async def decide(self, body):
        self.called += 1
        self.last = body
        answers = {}
        for name, q in body["questions"].items():
            keys = list(q.get("criteria") or [])
            if q.get("type") == "multi":
                answers[name] = {"type": "multi", "selected": keys[:1], "probabilities": {k: 0.9 for k in keys},
                                 "threshold": 0.5, "set_confidence": 0.9}
            else:
                chosen = ocr.OTHER_KEY if self.pick_other else keys[0]
                answers[name] = {"type": "choice", "choice": chosen, "confidence": 0.9,
                                 "probabilities": {k: 0.9 / len(keys) for k in keys}}
        return {"model": self.name, "answers": answers,
                "usage": {"questions": len(answers), "branches": len(answers), "input_tokens": 7,
                          "output_tokens": 0, "latency_ms": 1.0}}


MEDIA = {"type": "image", "name": "faktura.png", "ocr": "on", "data": "encoded-image"}


def question(regex=r"\d+[,.]\d{2}", kind="choice", **kwargs):
    return dict(type=kind, instructions="Jaka jest kwota faktury?", generated_criteria={
        "preprocessor": "ocr", "source": "faktura.png", "filter": regex}, **kwargs)


def request(questions):
    return {"state": "", "lang": "pl", "media": [dict(MEDIA)], "questions": questions}


def run(fake, body):
    return asyncio.run(fake.serve(body))


def test_static_and_ocr_merge_deduplicates_and_tracks_origin():
    fake = Fake(["VAT 12,00", "suma 15,00", "kopia 12,00"])
    body = request({"kwota": question(criteria=["12,00", "99,00"])})
    unchanged = copy.deepcopy(body)
    out = run(fake, body)
    assert body == unchanged
    assert list(fake.last["questions"]["kwota"]["criteria"]) == ["12,00", "99,00", "15,00", ocr.OTHER_KEY]
    assert out["answers"]["kwota"]["choice"] == "12,00"
    assert out["answers"]["kwota"]["source"] == ["static", "ocr"]
    assert out["usage"]["ocr"] == {"images": 1, "matches": {"kwota": 3}}
    assert fake.ocr_client.calls == [("encoded-image", 0)]


def test_single_ocr_candidate_always_invokes_decision_model():
    fake = Fake(["Wartość: 30,00"])
    out = run(fake, request({"kwota": question()}))
    assert fake.called == 1
    assert list(fake.last["questions"]["kwota"]["criteria"]) == ["30,00", ocr.OTHER_KEY]
    assert out["answers"]["kwota"]["choice"] == "30,00"
    assert out["answers"]["kwota"]["source"] == ["ocr"]


def test_other_selected_is_null_and_has_probability_entry():
    fake = Fake(["30,00"], other=True)
    out = run(fake, request({"kwota": question(__other__="Żadna wartość")}))
    ans = out["answers"]["kwota"]
    assert ans["choice"] is None and ans["__other__"] is True and ans["source"] == ["__other__"]
    assert ocr.OTHER_KEY in ans["probabilities"]
    opts = to_items("", fake.last["questions"], max_opts=10, force_lang="pl")[0]["options"]
    assert opts == ["30,00", "Żadna wartość"]


def test_default_other_label_is_readable():
    fake = Fake(["30,00"])
    run(fake, request({"q": question()}))
    opts = to_items("", fake.last["questions"], max_opts=10, force_lang="pl")[0]["options"]
    assert opts == ["30,00", "Żadna z powyższych"]


def test_plain_static_choice_unchanged():
    fake = Fake(["30,00"])
    body = {"state": "", "questions": {"q": {"type": "choice", "criteria": ["A", "B"]}}}
    out = run(fake, body)
    assert list(fake.last["questions"]["q"]["criteria"]) == ["A", "B"]
    assert out["answers"]["q"]["choice"] == "A"
    assert "preprocessor_only" not in out["usage"]
    assert fake.ocr_client.calls == []


def test_explicit_other_with_static_criteria_only():
    fake = Fake([])
    out = run(fake, {"state": "", "questions": {
        "q": {"type": "choice", "criteria": ["A", "B"], "__other__": "Żadna"}}})
    assert list(fake.last["questions"]["q"]["criteria"]) == ["A", "B", ocr.OTHER_KEY]
    assert to_items("", fake.last["questions"], max_opts=10)[0]["options"][-1] == "Żadna"
    assert out["answers"]["q"]["choice"] == "A"


def test_multi_merges_without_adding_other():
    fake = Fake(["VAT: 20,00"])
    out = run(fake, request({"kwota": question(kind="multi", criteria=["10,00"])}))
    assert list(fake.last["questions"]["kwota"]["criteria"]) == ["10,00", "20,00"]
    assert ocr.OTHER_KEY not in fake.last["questions"]["kwota"]["criteria"]
    assert out["answers"]["kwota"]["selected"] == ["10,00"]
    assert out["answers"]["kwota"]["sources"] == {"10,00": ["static"]}


def test_zero_candidates_without_static_values_avoids_inference():
    fake = Fake(["Nie ma kwot"])
    out = run(fake, request({"kwota": question()}))
    assert fake.called == 0
    assert out["answers"]["kwota"] == {"type": "choice", "choice": None,
                                       "confidence": None, "probabilities": {},
                                       "decision_source": "ocr_preprocessor", "reason": "no_candidates"}
    assert out["usage"]["input_tokens"] == 0
    assert out["usage"]["preprocessor_only"] == 1


def test_zero_matches_but_static_criteria_runs_inference():
    fake = Fake(["Nie ma kwot"])
    out = run(fake, request({"q": question(criteria=["A"])}))
    assert fake.called == 1
    assert list(fake.last["questions"]["q"]["criteria"]) == ["A", ocr.OTHER_KEY]
    assert out["answers"]["q"]["choice"] == "A"


def test_empty_multi_returns_empty_set_without_inference():
    fake = Fake(["Nie ma kwot"])
    out = run(fake, request({"q": question(kind="multi")}))
    assert fake.called == 0
    assert out["answers"]["q"]["selected"] == []
    assert out["answers"]["q"]["set_confidence"] is None


def test_empty_question_does_not_prevent_other_question_inference():
    fake = Fake(["Brak kwot"])
    out = run(fake, request({
        "missing": question(),
        "normal": {"type": "choice", "criteria": ["A", "B"]}}))
    assert fake.called == 1
    assert list(out["answers"]) == ["missing", "normal"]
    assert out["answers"]["missing"]["choice"] is None
    assert out["answers"]["normal"]["choice"] == "A"
    assert out["usage"]["questions"] == 2 and out["usage"]["branches"] == 1


def test_same_image_read_once_for_multiple_questions():
    fake = Fake(["30,00", "Data 12/11/2026"])
    run(fake, request({"a": question(), "b": question(regex=r"\d{2}/\d{2}/\d{4}")}))
    assert len(fake.ocr_client.calls) == 1


@pytest.mark.parametrize("q,media,match", [
    (question(regex="["), MEDIA, "invalid OCR regex"),
    (question(), dict(MEDIA, ocr="off"), "source must name"),
    (question(), dict(MEDIA, type="audio"), "only for images"),
    (question(), dict(MEDIA, name="elsewhere"), "source must name"),
    ({"type": "score", "generated_criteria": question()["generated_criteria"]}, MEDIA, "supports choice and multi"),
    ({"type": "choice", "generated_criteria": {"preprocessor": "unknown",
                                               "source": "faktura.png", "filter": "x"}}, MEDIA, "preprocessor"),
    (question(criteria=[ocr.OTHER_KEY]), MEDIA, "reserved"),
    (question(__other__=123), MEDIA, "__other__ must be"),
])
def test_invalid_specs_rejected(q, media, match):
    fake = Fake(["30,00"])
    with pytest.raises(RequestError, match=match):
        run(fake, {"state": "", "media": [media], "questions": {"q": q}})


def test_missing_service_configuration_rejected():
    fake = Fake([], configured=False)
    with pytest.raises(RequestError, match="--ocr-api"):
        run(fake, request({"q": question()}))


def test_more_than_model_limit_rejected():
    fake = Fake(["1,00 2,00 3,00"], max_options=3)
    with pytest.raises(RequestError, match="candidate limit"):
        run(fake, request({"q": question()}))


def test_duplicate_ocr_source_names_rejected():
    fake = Fake(["1,00"])
    body = request({"q": question()})
    body["media"].append(dict(MEDIA))
    with pytest.raises(RequestError, match="duplicate"):
        run(fake, body)


def test_cli_has_separate_service_and_ocr_addresses():
    a = parser().parse_args(["--ocr-api", "paddlex", "--ocr-url", "http://localhost:8091",
                             "--ocr-timeout", "2", "--host", "127.0.0.1", "--port", "8000"])
    assert a.ocr_api == "paddlex" and a.ocr_url == "http://localhost:8091"
    assert a.port == 8000 and a.ocr_timeout == 2


def _image_base64():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (5, 5), "white").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_paddlex_http_contract(monkeypatch):
    got = []
    def handler(req):
        assert req.url.path == "/ocr"
        got.append(json.loads(req.content))
        return httpx.Response(200, json={"errorCode": 0, "result": {"ocrResults": [
            {"prunedResult": {"rec_texts": ["VAT 23%", "Kwota 120,00"], "rec_scores": [0.9, 0.98]}}]}})
    original = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(ocr.httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs))
    lines = asyncio.run(ocr.OCRClient("paddlex", "http://localhost:8080").read(_image_base64(), 0))
    assert lines == ["VAT 23%", "Kwota 120,00"]
    assert got[0]["fileType"] == 1 and got[0]["visualize"] is False
    assert base64.b64decode(got[0]["file"]).startswith(b"\x89PNG")


def test_paddlex_error_is_gateway_error_not_null(monkeypatch):
    original = httpx.AsyncClient
    monkeypatch.setattr(ocr.httpx, "AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(
                            lambda req: httpx.Response(503, text="unavailable")), **kwargs))
    with pytest.raises(ocr.OCRServiceError) as ex:
        asyncio.run(ocr.OCRClient("paddlex", "http://localhost:8080").read(_image_base64(), 0))
    body, status = error_body(ex.value)
    assert status == 502 and body["code"] == "UPSTREAM_OCR_ERROR"


def test_pdf_refused_before_http_call():
    with pytest.raises(RequestError, match="not a readable image"):
        asyncio.run(ocr.OCRClient("paddlex", "http://localhost:8080").read(
            base64.b64encode(b"%PDF-1.7\nno images").decode(), 0))


def test_http_endpoint_runs_ocr_and_preserves_choice_shape():
    from basal.server import http_app

    fake = Fake(["Kwota 99,00"])
    app = http_app(fake, "eager")

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://basal.test") as client:
            return await client.post("/v1/systemone", json=request({"q": question()}))

    response = asyncio.run(call())
    assert response.status_code == 200
    body = response.json()
    assert body["answers"]["q"]["choice"] == "99,00"
    assert body["answers"]["q"]["source"] == ["ocr"]
    assert body["usage"]["ocr"]["images"] == 1


def test_http_endpoint_reports_ocr_failure_as_502():
    from basal.server import http_app

    class FailingReader:
        async def read(self, *args):
            raise ocr.OCRServiceError("OCR unavailable")

    fake = Fake([])
    fake.ocr_client = FailingReader()
    app = http_app(fake, "eager")

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://basal.test") as client:
            return await client.post("/v1/systemone", json=request({"q": question()}))

    response = asyncio.run(call())
    assert response.status_code == 502
    assert response.json()["code"] == "UPSTREAM_OCR_ERROR"


def test_multi_can_exceed_choice_option_limit():
    fake = Fake(["1 2 3 4"], max_options=3)
    out = run(fake, request({"q": question(regex=r"\d", kind="multi")}))
    assert fake.called == 1
    assert list(fake.last["questions"]["q"]["criteria"]) == ["1", "2", "3", "4"]
    assert out["answers"]["q"]["selected"] == ["1"]


def test_static_description_deduplicates_against_ocr():
    fake = Fake(["A: 15,00 B: 25,00"])
    out = run(fake, request({"q": question(criteria={"gross": "15,00"})}))
    assert list(fake.last["questions"]["q"]["criteria"]) == ["gross", "25,00", ocr.OTHER_KEY]
    assert out["answers"]["q"]["source"] == ["static", "ocr"]


# --- Per-image media.context: original image (default), OCR text, or both ---

def test_context_text_replaces_image_and_injects_ocr_without_mutating_request():
    fake = Fake(["FAKTURA VAT", "Kwota brutto: 1250,00 PLN"])
    body = request({"kwota": question(criteria=["999,00"])})
    body["state"] = "Odczytaj fakturę."
    body["media"][0]["context"] = "text"
    original = copy.deepcopy(body)

    out = run(fake, body)

    assert body == original
    assert fake.called == 1
    assert "media" not in fake.last  # the text-only backend must not receive an image
    assert fake.last["state"] == (
        'Odczytaj fakturę.\n\n[OCR: "faktura.png"]\n'
        "FAKTURA VAT\nKwota brutto: 1250,00 PLN\n[/OCR]"
    )
    assert list(fake.last["questions"]["kwota"]["criteria"]) == ["999,00", "1250,00", ocr.OTHER_KEY]
    assert out["usage"]["ocr"] == {"images": 1, "matches": {"kwota": 1}}
    assert fake.ocr_client.calls == [("encoded-image", 0)]
    assert Server.media_of(fake, fake.last, None) is None


def test_context_text_works_with_static_questions_and_no_generated_criteria():
    fake = Fake(["VAT 23%", "KWOTA: 1250,00"])
    body = request({"decide": {"type": "choice", "instructions": "Czy jest faktura?",
                               "criteria": ["tak", "nie"]}})
    body["media"][0]["context"] = "text"
    out = run(fake, body)
    assert fake.called == 1
    assert fake.last["questions"] == body["questions"]
    assert list(fake.last["questions"]["decide"]["criteria"]) == ["tak", "nie"]
    assert "media" not in fake.last
    assert "VAT 23%\nKWOTA: 1250,00" in fake.last["state"]
    assert out["usage"]["ocr"] == {"images": 1, "matches": {}}
    assert out["usage"]["preprocessor_only"] == 0
    assert ocr.OTHER_KEY not in out["answers"]["decide"]["probabilities"]
    assert len(fake.ocr_client.calls) == 1


def test_context_image_is_default_and_keeps_original_media():
    for explicit in (False, True):
        fake = Fake(["Kwota brutto: 1250,00"])
        body = request({"q": question()})
        if explicit:
            body["media"][0]["context"] = "image"
        out = run(fake, body)
        assert fake.last["media"] == body["media"]
        assert fake.last["state"] == ""
        assert len(fake.ocr_client.calls) == 1
        assert out["usage"]["ocr"]["images"] == 1


def test_explicit_image_context_without_generated_does_not_run_ocr():
    fake = Fake(["unused"])
    body = request({"q": {"type": "choice", "criteria": ["tak", "nie"]}})
    body["media"][0]["context"] = "image"
    out = run(fake, body)
    assert fake.ocr_client.calls == []
    assert fake.last is body
    assert out["usage"].get("ocr") is None


def test_context_both_preserves_image_and_adds_text():
    fake = Fake(["Kwota brutto 99,00"])
    body = request({"q": question()})
    body["media"][0]["context"] = "both"
    out = run(fake, body)
    assert fake.last["media"] == body["media"]
    assert '[OCR: "faktura.png"]\nKwota brutto 99,00\n[/OCR]' in fake.last["state"]
    assert out["usage"]["ocr"]["images"] == 1
    assert len(fake.ocr_client.calls) == 1


def test_context_both_works_without_generated_candidates():
    fake = Fake(["Treść dokumentu"])
    body = request({"q": {"type": "choice", "criteria": ["A", "B"]}})
    body["media"][0]["context"] = "both"
    out = run(fake, body)
    assert fake.last["media"] == body["media"]
    assert "Treść dokumentu" in fake.last["state"]
    assert fake.last["questions"]["q"] == body["questions"]["q"]
    assert len(fake.ocr_client.calls) == 1
    assert out["usage"]["ocr"] == {"images": 1, "matches": {}}


def test_context_text_and_both_inject_in_media_order_and_retain_remaining_media():
    fake = Fake(["Tekst z dokumentu"])
    body = request({"q": {"type": "choice", "criteria": ["A", "B"]}})
    body["media"] = [
        dict(MEDIA, context="text"),
        dict(MEDIA, name="drugi.png", data="second-image", context="both"),
        {"type": "image", "name": "trzeci.png", "data": "third-image"},
    ]
    run(fake, body)
    assert fake.last["media"] == body["media"][1:]
    assert fake.last["state"].index('OCR: "faktura.png"') < fake.last["state"].index('OCR: "drugi.png"')
    assert 'OCR: "trzeci.png"' not in fake.last["state"]
    assert fake.ocr_client.calls == [("encoded-image", 0), ("second-image", 1)]


def test_multi_questions_share_single_read_for_text_context_and_candidates():
    fake = Fake(["Kwota 1250,00", "VAT 1250,00"])
    body = request({"q1": question(), "q2": question(kind="multi")})
    body["media"][0]["context"] = "text"
    out = run(fake, body)
    assert len(fake.ocr_client.calls) == 1
    assert "media" not in fake.last
    assert out["usage"]["ocr"]["images"] == 1
    assert out["usage"]["ocr"]["matches"] == {"q1": 2, "q2": 2}


def test_context_text_with_structured_state_uses_json_representation():
    fake = Fake(["OCR L1"])
    body = request({"q": {"type": "choice", "criteria": ["A", "B"]}})
    body["state"] = {"identyfikator": 1, "zweryfikuj": True}
    body["media"][0]["context"] = "text"
    run(fake, body)
    assert fake.last["state"].startswith('{"identyfikator": 1, "zweryfikuj": true}')
    assert "[OCR:" in fake.last["state"]
    assert isinstance(body["state"], dict)


def test_context_text_with_no_recognized_text_still_runs_static_decision():
    fake = Fake([])
    body = request({"q": {"type": "choice", "criteria": ["A", "B"]}})
    body["media"][0]["context"] = "text"
    out = run(fake, body)
    assert fake.called == 1
    assert "media" not in fake.last
    assert "[/OCR]" in fake.last["state"]
    assert out["usage"]["ocr"]["images"] == 1


@pytest.mark.parametrize("context", ["text", "both"])
@pytest.mark.parametrize("flag", [None, "off", False])
def test_context_requires_ocr_on(context, flag):
    fake = Fake([])
    body = request({"q": {"type": "choice", "criteria": ["A", "B"]}})
    body["media"][0]["context"] = context
    if flag is None:
        del body["media"][0]["ocr"]
    else:
        body["media"][0]["ocr"] = flag
    with pytest.raises(RequestError, match="requires ocr"):
        run(fake, body)
    assert fake.ocr_client.calls == []


@pytest.mark.parametrize("context", [None, "", "TEXT", "auto", False, 1])
def test_invalid_context_is_rejected(context):
    fake = Fake([])
    body = request({"q": {"type": "choice", "criteria": ["A", "B"]}})
    body["media"][0]["context"] = context
    with pytest.raises(RequestError, match="context must be"):
        run(fake, body)
    assert fake.ocr_client.calls == []


def test_context_rejected_for_audio():
    fake = Fake([])
    body = request({"q": {"type": "choice", "criteria": ["A", "B"]}})
    body["media"][0].update(type="audio", context="text")
    with pytest.raises(RequestError, match="only for images"):
        run(fake, body)


@pytest.mark.parametrize("context", ["text", "both"])
def test_context_requires_configured_ocr_service(context):
    fake = Fake([], configured=False)
    body = request({"q": {"type": "choice", "criteria": ["A", "B"]}})
    body["media"][0]["context"] = context
    with pytest.raises(RequestError, match="--ocr-api"):
        run(fake, body)


def test_context_and_images_alias_are_mutually_exclusive():
    fake = Fake([])
    body = request({"q": {"type": "choice", "criteria": ["A", "B"]}})
    body["media"][0]["context"] = "text"
    body["images"] = ["data:image/png;base64,abc"]
    with pytest.raises(RequestError, match="media or images"):
        run(fake, body)


def test_context_ocr_upstream_failure_returns_502_without_generated_criteria():
    from basal.server import http_app

    class FailingReader:
        async def read(self, *args):
            raise ocr.OCRServiceError("OCR unavailable")

    fake = Fake([])
    fake.ocr_client = FailingReader()
    app = http_app(fake, "eager")
    body = request({"q": {"type": "choice", "criteria": ["A", "B"]}})
    body["media"][0]["context"] = "text"

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://basal.test") as client:
            return await client.post("/v1/systemone", json=body)

    response = asyncio.run(call())
    assert response.status_code == 502
    assert response.json()["code"] == "UPSTREAM_OCR_ERROR"
    assert fake.called == 0


def test_context_default_image_without_generated_ignores_unavailable_ocr():
    fake = Fake([], configured=False)
    body = request({"q": {"type": "choice", "criteria": ["A", "B"]}})
    body["media"][0]["context"] = "image"
    assert run(fake, body)["answers"]["q"]["choice"] == "A"


# --- media.context="none": OCR candidates only; no image or OCR text in Basal state ---

def test_context_none_generated_choice_supplies_only_candidate_options_to_model():
    fake = Fake(["Kwota brutto: 1250,00", "Data 12/11/2026"])
    body = request({"q": question()})
    body["state"] = "Sprawdź kwotę brutto."
    body["media"][0]["context"] = "none"
    original = copy.deepcopy(body)

    result = run(fake, body)

    assert body == original
    assert fake.called == 1
    assert "media" not in fake.last
    assert fake.last["state"] == "Sprawdź kwotę brutto."
    assert list(fake.last["questions"]["q"]["criteria"]) == ["1250,00", ocr.OTHER_KEY]
    assert "1250,00" not in fake.last["state"]
    assert result["answers"]["q"]["choice"] == "1250,00"
    assert result["answers"]["q"]["source"] == ["ocr"]
    assert result["usage"]["ocr"] == {"images": 1, "matches": {"q": 1}}
    assert fake.ocr_client.calls == [("encoded-image", 0)]
    assert Server.media_of(fake, fake.last, None) is None


def test_context_none_dynamic_multi_uses_extracted_labels_without_media():
    fake = Fake(["20,00 / 30,00"])
    body = request({"q": question(kind="multi")})
    body["media"][0]["context"] = "none"
    result = run(fake, body)

    assert fake.called == 1
    assert "media" not in fake.last
    assert fake.last["state"] == ""
    assert list(fake.last["questions"]["q"]["criteria"]) == ["20,00", "30,00"]
    assert result["answers"]["q"]["sources"] == {"20,00": ["ocr"]}
    assert ocr.OTHER_KEY not in fake.last["questions"]["q"]["criteria"]


@pytest.mark.parametrize("enabled", ["on", "off", None])
def test_context_none_static_only_drops_image_without_calling_ocr(enabled):
    fake = Fake(["unused"], configured=False)
    body = request({"q": {"type": "choice", "criteria": ["A", "B"]}})
    body["state"] = "Oryginalny stan."
    body["media"][0]["context"] = "none"
    if enabled is None:
        del body["media"][0]["ocr"]
    else:
        body["media"][0]["ocr"] = enabled
    result = run(fake, body)

    assert fake.called == 1
    assert "media" not in fake.last
    assert fake.last["state"] == body["state"]
    assert fake.last["questions"] == body["questions"]
    assert "ocr" not in result["usage"]
    assert "preprocessor_only" not in result["usage"]


def test_context_none_generated_requires_ocr_on():
    fake = Fake(["unused"])
    body = request({"q": question()})
    body["media"][0].update(context="none", ocr="off")
    with pytest.raises(RequestError, match="source must name"):
        run(fake, body)
    assert fake.ocr_client.calls == []


def test_context_none_and_both_share_ocr_and_keep_only_both_image():
    fake = Fake(["Kwota: 1250,00"])
    body = request({"q": question()})
    body["media"] = [
        dict(MEDIA, context="none"),
        dict(MEDIA, name="kopia.png", context="both", data="second-image"),
        {"type": "audio", "data": "audio-base64"},
    ]
    result = run(fake, body)

    assert fake.last["media"] == body["media"][1:]
    assert '[OCR: "kopia.png"]' in fake.last["state"]
    assert "faktura.png" not in fake.last["state"]
    assert "Kwota: 1250,00" in fake.last["state"]
    assert fake.ocr_client.calls == [("encoded-image", 0), ("second-image", 1)]
    assert result["usage"]["ocr"]["images"] == 2


def test_context_none_and_text_drops_both_images_and_injects_text_only_for_text():
    fake = Fake(["Kwota: 1250,00"])
    body = request({"q": question()})
    body["media"] = [
        dict(MEDIA, context="none"),
        dict(MEDIA, name="drugi.png", context="text", data="second-image"),
    ]
    result = run(fake, body)

    assert "media" not in fake.last
    assert '[OCR: "drugi.png"]' in fake.last["state"]
    assert "faktura.png" not in fake.last["state"]
    assert fake.ocr_client.calls == [("encoded-image", 0), ("second-image", 1)]
    assert result["usage"]["ocr"]["images"] == 2


def test_context_none_no_matches_skips_model_and_returns_null_choice():
    fake = Fake(["Brak kwoty"])
    body = request({"q": question()})
    body["media"][0]["context"] = "none"
    result = run(fake, body)
    assert fake.called == 0
    assert result["answers"]["q"]["choice"] is None
    assert result["answers"]["q"]["reason"] == "no_candidates"
    assert result["usage"]["ocr"]["images"] == 1


def test_context_none_ocr_failure_returns_502():
    from basal.server import http_app

    class FailingReader:
        async def read(self, *args):
            raise ocr.OCRServiceError("OCR unavailable")

    fake = Fake([])
    fake.ocr_client = FailingReader()
    body = request({"q": question()})
    body["media"][0]["context"] = "none"
    app = http_app(fake, "eager")

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://basal.test") as client:
            return await client.post("/v1/systemone", json=body)

    response = asyncio.run(call())
    assert response.status_code == 502
    assert response.json()["code"] == "UPSTREAM_OCR_ERROR"
    assert fake.called == 0


def test_context_none_explicit_static_fallback_only_drops_image_without_ocr_call():
    fake = Fake([], configured=False)
    body = request({"q": {"type": "choice", "criteria": ["A", "B"], "__other__": "Żadne"}})
    body["media"][0]["context"] = "none"
    result = run(fake, body)
    assert "media" not in fake.last
    assert list(fake.last["questions"]["q"]["criteria"]) == ["A", "B", ocr.OTHER_KEY]
    assert "ocr" not in result["usage"]
    assert result["usage"]["preprocessor_only"] == 0


# Reserved OCR choice option: the same __other__ identifier in every API position.

def test_legacy_other_request_field_rejected_not_silently_ignored():
    fake = Fake([], configured=False)
    body = {"state": "Example", "questions": {
        "q": {"type": "choice", "criteria": ["A", "B"], "other": "No match"}}}
    with pytest.raises(RequestError, match="use __other__ instead of other"):
        run(fake, body)
    assert fake.called == 0


def test_legacy_other_rejected_with_ocr_source():
    fake = Fake(["30,00"])
    body = request({"q": question(other="No match")})
    with pytest.raises(RequestError, match="use __other__ instead of other"):
        run(fake, body)
    assert fake.ocr_client.calls == []


def test_explicit_other_selected_without_generated_candidates():
    fake = Fake([], other=True, configured=False)
    body = {"state": "Example", "questions": {
        "q": {"type": "choice", "criteria": ["A", "B"], "__other__": "No match"}}}
    out = run(fake, body)
    ans = out["answers"]["q"]
    assert ans["choice"] is None
    assert ans["__other__"] is True
    assert ans["source"] == ["__other__"]
    assert list(ans["probabilities"]) == ["A", "B", "__other__"]
    assert "__none__" not in ans["probabilities"]
    assert "other" not in ans
    assert to_items("", fake.last["questions"], max_opts=10)[0]["options"][-1] == "No match"


def test_explicit_null_other_uses_default_label():
    fake = Fake([])
    body = {"state": "", "lang": "pl", "questions": {
        "q": {"type": "choice", "criteria": ["A", "B"], "__other__": None}}}
    run(fake, body)
    opts = to_items("", fake.last["questions"], max_opts=10, force_lang="pl")[0]["options"]
    assert opts[-1] == "Żadna z powyższych"


def test_reserved_other_cannot_be_used_as_static_criterion():
    fake = Fake([])
    body = {"state": "", "questions": {"q": {
        "type": "choice", "criteria": ["A", "__other__"], "__other__": "No match"}}}
    with pytest.raises(RequestError, match="reserved"):
        run(fake, body)


def test_ocr_match_equal_to_reserved_other_never_becomes_ordinary_option():
    fake = Fake(["__other__ A"])
    body = request({"q": question(regex=r"__other__|A")})
    out = run(fake, body)
    assert list(fake.last["questions"]["q"]["criteria"]) == ["A", "__other__"]
    assert out["usage"]["ocr"]["matches"]["q"] == 2
    assert out["answers"]["q"]["source"] == ["ocr"]


def test_reserved_other_field_rejected_on_multi():
    fake = Fake([])
    body = {"state": "", "questions": {"q": {
        "type": "multi", "criteria": ["A", "B"], "__other__": "No match"}}}
    with pytest.raises(RequestError, match="__other__ only applies to choice"):
        run(fake, body)


def test_ordinary_criterion_named_other_is_unaffected():
    fake = Fake([])
    body = {"state": "", "questions": {"q": {
        "type": "choice", "criteria": ["other", "A"]}}}
    out = run(fake, body)
    assert out["answers"]["q"]["choice"] == "other"
    assert "__other__" not in fake.last["questions"]["q"]
    assert "ocr" not in out["usage"]


def test_http_choice_other_returns_canonical_reserved_key():
    from basal.server import http_app

    fake = Fake(["30,00"], other=True)
    app = http_app(fake, "eager")
    body = request({"q": question(__other__="Nie ma poprawnej kwoty")})

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://basal.test") as client:
            return await client.post("/v1/systemone", json=body)

    res = asyncio.run(call())
    assert res.status_code == 200
    ans = res.json()["answers"]["q"]
    assert ans["choice"] is None
    assert ans["__other__"] is True
    assert ans["source"] == ["__other__"]
    assert "__other__" in ans["probabilities"]
    assert "__none__" not in ans["probabilities"]


def test_http_legacy_other_field_returns_422():
    from basal.server import http_app

    fake = Fake([])
    app = http_app(fake, "eager")
    body = {"state": "", "questions": {"q": {
        "type": "choice", "criteria": ["A", "B"], "other": "Not applicable"}}}

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://basal.test") as client:
            return await client.post("/v1/systemone", json=body)

    res = asyncio.run(call())
    assert res.status_code == 422
    assert "__other__" in res.text


def test_reserved_other_key_rejected_in_plain_choice_without_preprocessor():
    q = {"q": {"type": "choice", "instructions": "Which?",
               "criteria": ["A", "__other__"]}}
    with pytest.raises(RequestError, match="reserved"):
        to_items("state", q, max_opts=10)


def test_other_literal_remains_regular_criterion_in_plain_choice():
    q = {"q": {"type": "choice", "instructions": "Which?",
               "criteria": ["A", "other"]}}
    items = to_items("state", q, max_opts=10)
    assert items[0]["keys"] == ["A", "other"]


# --- OCR security and resource-limit regressions ---

@pytest.mark.parametrize("context", ["text", "none", "both", "image"])
def test_media_cap_enforced_before_any_ocr_calls(context):
    fake = Fake(["Kwota 10,00"])
    body = request({"q": question()})
    body["media"] = [
        dict(MEDIA, name="faktura.png" if i == 0 else f"doc-{i}.png",
             context=context) for i in range(ocr.media_module.MAX_ITEMS + 1)
    ]
    with pytest.raises(RequestError, match="at most 8"):
        run(fake, body)
    assert fake.ocr_client.calls == []
    assert fake.called == 0


def test_media_cap_applies_when_none_drops_all_images_without_ocr():
    fake = Fake([], configured=False)
    body = {"state": "test", "media": [
        {"type": "image", "data": "unused", "context": "none"}
        for _ in range(ocr.media_module.MAX_ITEMS + 1)],
        "questions": {"q": {"type": "choice", "criteria": ["A", "B"]}}}
    with pytest.raises(RequestError, match="at most 8"):
        run(fake, body)
    assert fake.called == 0


def test_media_cap_allows_eight_text_ocr_images():
    fake = Fake(["faktura"])
    body = {"state": "test", "media": [
        dict(MEDIA, name=f"doc-{i}.png", context="text")
        for i in range(ocr.media_module.MAX_ITEMS)],
        "questions": {"q": {"type": "choice", "criteria": ["A", "B"]}}}
    out = run(fake, body)
    assert fake.called == 1
    assert len(fake.ocr_client.calls) == ocr.media_module.MAX_ITEMS
    assert out["usage"]["ocr"]["images"] == ocr.media_module.MAX_ITEMS


@pytest.mark.parametrize("bad_image", [
    "%%%invalid-base64%%%",
    base64.b64encode(b"%PDF-1.7\nno raster image").decode(),
    base64.b64encode(b"no readable image").decode(),
])
def test_invalid_ocr_images_return_http_422_without_contacting_paddlex(bad_image):
    from basal.server import http_app
    fake = Fake([])
    fake.ocr_client = ocr.OCRClient("paddlex", "http://unreachable.invalid:8080")
    app = http_app(fake, "eager")
    body = request({"q": question()})
    body["media"][0]["data"] = bad_image

    async def call():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://basal.test") as client:
            return await client.post("/v1/systemone", json=body)

    res = asyncio.run(call())
    assert res.status_code == 422, res.text
    assert res.json()["code"] == "INVALID_REQUEST"
    assert fake.called == 0


def test_oversized_base64_and_pixel_count_are_client_errors(monkeypatch):
    monkeypatch.setattr(ocr.media_module, "MAX_BYTES", 5)
    with pytest.raises(RequestError, match="more than"):
        asyncio.run(ocr.OCRClient("paddlex", "http://unused").read(_image_base64(), 0))
    monkeypatch.setattr(ocr.media_module, "MAX_BYTES", 20 * 1024 * 1024)
    monkeypatch.setattr(ocr.media_module, "MAX_PIXELS", 20)
    from PIL import Image
    with pytest.warns(Image.DecompressionBombWarning):
        with pytest.raises(RequestError, match="pixels"):
            asyncio.run(ocr.OCRClient("paddlex", "http://unused").read(_image_base64(), 0))


def _mock_paddlex(monkeypatch, handler):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        ocr.httpx, "AsyncClient",
        lambda **kwargs: original(transport=transport, **kwargs),
    )


def test_ocr_content_length_limit_checked_before_body_is_read(monkeypatch):
    monkeypatch.setattr(ocr, "MAX_OCR_RESPONSE_BYTES", 100)
    chunks_read = []

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            chunks_read.append(True)
            yield b"{}"

    def handler(req):
        assert req.headers["accept-encoding"] == "identity"
        return httpx.Response(200, headers={"Content-Length": "100000000"},
                              stream=Stream())

    _mock_paddlex(monkeypatch, handler)
    with pytest.raises(ocr.OCRServiceError, match="exceeds"):
        asyncio.run(ocr.OCRClient("paddlex", "http://localhost:8080").read(_image_base64(), 0))
    assert chunks_read == []


def test_ocr_unknown_length_body_is_bounded_while_streaming(monkeypatch):
    monkeypatch.setattr(ocr, "MAX_OCR_RESPONSE_BYTES", 100)

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(10):
                yield b"X" * 40

    _mock_paddlex(monkeypatch, lambda _: httpx.Response(200, stream=Stream()))
    with pytest.raises(ocr.OCRServiceError, match="exceeds") as ex:
        asyncio.run(ocr.OCRClient("paddlex", "http://localhost:8080").read(_image_base64(), 0))
    assert error_body(ex.value)[1] == 502


def test_ocr_rejects_compressed_upstream_response_before_decompression(monkeypatch):
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"invalid gzip"

    _mock_paddlex(monkeypatch, lambda _: httpx.Response(
        200, headers={"Content-Encoding": "gzip"}, stream=Stream()))
    with pytest.raises(ocr.OCRServiceError, match="not be compressed"):
        asyncio.run(ocr.OCRClient("paddlex", "http://localhost:8080").read(_image_base64(), 0))


@pytest.mark.parametrize("raw", [
    b"bad JSON",
    b"[" * 1500 + b"]" * 1500,  # json.loads raises RecursionError, not ValueError
])
def test_upstream_invalid_or_deeply_nested_json_is_502(monkeypatch, raw):
    _mock_paddlex(monkeypatch, lambda _: httpx.Response(200, content=raw))
    with pytest.raises(ocr.OCRServiceError):
        asyncio.run(ocr.OCRClient("paddlex", "http://localhost:8080").read(_image_base64(), 0))


def test_ocr_wall_clock_timeout_despite_continuous_stream(monkeypatch):
    class Trickle(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(200):
                await asyncio.sleep(0.01)  # never exceeds per-read httpx timeout
                yield b" "

    _mock_paddlex(monkeypatch, lambda _: httpx.Response(200, stream=Trickle()))
    import time
    started = time.monotonic()
    with pytest.raises(ocr.OCRServiceError, match="TimeoutError") as ex:
        asyncio.run(ocr.OCRClient("paddlex", "http://localhost:8080",
                                  timeout=0.06).read(_image_base64(), 0))
    assert time.monotonic() - started < 0.8
    assert error_body(ex.value)[1] == 502


def test_ocr_wall_clock_deadline_also_includes_json_parsing(monkeypatch):
    import time
    original_loads = json.loads

    def slow_decode(payload):
        time.sleep(0.065)
        return original_loads(payload)

    _mock_paddlex(monkeypatch, lambda _: httpx.Response(200, content=b'{"errorCode":0}'))
    monkeypatch.setattr(ocr.json, "loads", slow_decode)
    with pytest.raises(ocr.OCRServiceError, match="TimeoutError"):
        asyncio.run(ocr.OCRClient("paddlex", "http://localhost:8080",
                                  timeout=0.04).read(_image_base64(), 0))


def test_failed_ocr_cancels_and_awaits_sibling_reads():
    class FailAndWait:
        def __init__(self):
            self.started_second = asyncio.Event()
            self.cancelled_second = False

        async def read(self, data, index):
            if index == 0:
                await self.started_second.wait()
                raise ocr.OCRServiceError("first OCR failed")
            self.started_second.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                self.cancelled_second = True
                raise

    fake = Fake([])
    fake.ocr_client = FailAndWait()
    body = {"state": "", "questions": {
        "q": {"type": "choice", "criteria": ["A", "B"]}},
        "media": [dict(MEDIA, name=f"image-{i}.png", context="text") for i in range(2)]}
    with pytest.raises(ocr.OCRServiceError, match="first OCR failed"):
        run(fake, body)
    assert fake.ocr_client.cancelled_second
    assert fake.called == 0


def test_cancelled_basal_request_cleans_up_all_ocr_tasks():
    class HangingReads:
        def __init__(self):
            self.started = asyncio.Event()
            self.started_count = 0
            self.cancelled_count = 0

        async def read(self, data, index):
            self.started_count += 1
            if self.started_count == 2:
                self.started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                self.cancelled_count += 1
                raise

    fake = Fake([])
    fake.ocr_client = HangingReads()
    body = {"state": "", "questions": {
        "q": {"type": "choice", "criteria": ["A", "B"]}},
        "media": [dict(MEDIA, name=f"image-{i}.png", context="text") for i in range(2)]}

    async def cancel_request():
        task = asyncio.create_task(fake.serve(body))
        await fake.ocr_client.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fake.ocr_client.cancelled_count == 2
        assert fake.in_flight == 0

    asyncio.run(cancel_request())
