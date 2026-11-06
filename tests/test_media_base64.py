"""Media data as base64 tools write it: wrapped every 76 characters (coreutils, MIME), the URL-safe alphabet with or
without its padding, an upper-case data: scheme. Each decodes to the same bytes as the plain form, which decodes as
before; a broken padding is still refused."""
import base64

import pytest

from basal import media

RAW = bytes([251, 255, 254, 250, 253, 252] * 40)  # its base64 has "+" and "/" (URL-safe: "-" and "_")


@pytest.mark.parametrize("wrap", [
    lambda s: s,
    lambda s: "\r\n".join(s[i:i + 76] for i in range(0, len(s), 76)),
    lambda s: s.replace("+", "-").replace("/", "_"),
    lambda s: "DATA:image/png;base64," + s,
], ids=["plain", "wrapped", "urlsafe", "upper-case-scheme"])
def test_base64_as_the_tools_write_it_is_accepted(wrap):
    assert media._raw(wrap(base64.b64encode(RAW).decode()), 0) == RAW


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_urlsafe_base64_without_its_padding_is_accepted(n):
    raw = RAW[: 240 + n]
    s = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    assert media._raw(s, 0) == raw and media._raw("data:image/png;base64," + s, 0) == raw
    assert media._raw(base64.urlsafe_b64encode(raw).decode(), 0) == raw  # with the padding, as before


def test_a_broken_padding_is_still_refused():
    # (not "QUJD=": a padding after a whole group is refused from Python 3.11 on but accepted by 3.10, as in 2.5.4)
    with pytest.raises(ValueError, match="not valid base64"):
        media._raw("QU=JD", 0)  # a padding inside the data
    with pytest.raises(ValueError, match="not valid base64"):
        media._raw("QUJDR", 0)  # 5 characters can never be base64
