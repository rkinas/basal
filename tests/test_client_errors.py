"""The Python client's errors name what happened: 401 / 403 an AuthError, a 503 of a server still starting (or a
router with no replica ready) NotReady, an out-of-memory 503 OutOfMemory; an OpenAI-style error body gives its
message (and type) as text. Both new classes are subclasses of the ones raised before (ServerError, OutOfMemory), so
existing handlers still catch them, and NotReady is retried as OutOfMemory is."""
import httpx
import pytest

from basal import client as C


def basal(responses, **kw):
    sent = []

    def handler(req):
        sent.append(req)
        status, body = responses[min(len(sent), len(responses)) - 1]
        return httpx.Response(status, json=body)
    return C.Basal("http://basal", http=httpx.Client(transport=httpx.MockTransport(handler)), **kw), sent


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors(status):
    b, _ = basal([(status, {"error": "missing or invalid API key", "code": "UNAUTHORIZED"})])
    with pytest.raises(C.AuthError) as e:
        b.yes_no("s", "Late?")
    assert isinstance(e.value, C.ServerError) and e.value.status == status
    assert str(e.value) == "missing or invalid API key"


def test_auth_error_in_the_openai_shape_gives_its_message():
    b, _ = basal([(401, {"error": {"message": "missing or invalid API key", "type": "authentication_error",
                                   "param": None, "code": "UNAUTHORIZED"}})])
    with pytest.raises(C.AuthError) as e:
        b.yes_no("s", "Late?")
    assert str(e.value) == "authentication_error: missing or invalid API key"


@pytest.mark.parametrize("body", [{"status": "starting"}, {"error": "no basal replica is ready"}])
def test_a_server_that_is_starting_is_not_ready_and_retried(body):
    ok = {"answers": {"q": {"type": "noul", "noul": 0.9}}}
    b, sent = basal([(503, body), (200, ok)], retries=1, backoff=0)
    assert b.yes_no("s", "Late?")["noul"] == 0.9 and len(sent) == 2
    b, _ = basal([(503, body)])
    with pytest.raises(C.NotReady) as e:
        b.yes_no("s", "Late?")
    assert isinstance(e.value, C.OutOfMemory) and e.value.status == 503


def test_out_of_memory_stays_out_of_memory():
    b, _ = basal([(503, {"error": "CUDA out of memory", "code": "OUT_OF_MEMORY"})])
    with pytest.raises(C.OutOfMemory) as e:
        b.yes_no("s", "Late?")
    assert not isinstance(e.value, C.NotReady) and str(e.value) == "CUDA out of memory"
