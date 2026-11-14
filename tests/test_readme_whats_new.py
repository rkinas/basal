"""README's release paragraphs ("X.Y.Z (date): ..."): one release head per paragraph, at its start; a paragraph links at
most its own release's WHATS_NEW section, once, at its end; and a link never follows a word directly. A generated
paragraph once got the previous release's link after every line ("any[WHATS_NEW](...260)."), spliced into the previous
paragraph."""
import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"
HEAD = re.compile(r"(\d+)\.(\d+)\.(\d+) \(\d{1,2} \w+ \d{4}\):")
LINK = re.compile(r"\[WHATS_NEW\]\(docs/WHATS_NEW\.md#what-is-new-in-basal-(\d+)\)")


def test_release_paragraphs_link_their_own_section_once():
    text = README.read_text(encoding="utf-8")
    for block in (b.strip() for b in re.split(r"\n\s*\n", text)):
        heads = HEAD.findall(block)
        if not heads:
            continue
        assert len(heads) == 1 and HEAD.match(block), (heads, block[:80])
        version, links = "".join(heads[0]), LINK.findall(block)
        assert links in ([], [version]), (version, links)
        assert not links or block.endswith(f"#what-is-new-in-basal-{version}).")
    assert not re.search(r"\S\[WHATS_NEW\]\(", text), re.search(r".{20}\S\[WHATS_NEW\]\(", text)
    versions = LINK.findall(text)
    assert len(versions) == len(set(versions)), versions
