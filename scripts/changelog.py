"""The release notes from fragments: every pull request adds its entry as docs/changes/<PR number>.md (one or more
list items, as they read in docs/WHATS_NEW.md), so open pull requests do not edit the same lines of WHATS_NEW.md.

  python scripts/changelog.py              the next release's section, from the fragments (nothing is written)
  python scripts/changelog.py 2.4.2        the section put at the top of docs/WHATS_NEW.md, the fragments removed
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGES, NOTES = ROOT / "docs" / "changes", ROOT / "docs" / "WHATS_NEW.md"


def fragments(changes=CHANGES):
    """[(PR number, entry text)] in the order of the PR numbers; a file not named <number>.md is an error (README.md
    aside), as is an empty one."""
    out = []
    for f in sorted(changes.glob("*.md")):
        if f.name == "README.md":
            continue
        if not re.fullmatch(r"\d+\.md", f.name):
            raise SystemExit(f"{f.name}: name it <PR number>.md")
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit(f"{f.name}: empty")
        out.append((int(f.stem), text))
    return sorted(out)


def section(version, entries):
    return f"# What is new in basal {version}\n\n" + "\n".join(text for _, text in entries) + "\n"


def release(version, changes=CHANGES, notes=NOTES):
    """The section of `version` at the top of `notes`, then the fragments removed. Returns the section."""
    entries = fragments(changes)
    if not entries:
        raise SystemExit(f"no fragments in {changes}")
    head = f"# What is new in basal {version}\n"
    old = notes.read_text(encoding="utf-8")
    if old.startswith(head) or f"\n{head}" in old:
        raise SystemExit(f"{notes.name} already has a section for {version}")
    new = section(version, entries)
    notes.write_text(new + "\n" + old, encoding="utf-8")
    for n, _ in entries:
        (changes / f"{n}.md").unlink()
    return new


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1 or argv and argv[0].startswith("-"):
        raise SystemExit(__doc__)
    if argv:
        release(argv[0])
        print(f"{NOTES.relative_to(ROOT)}: section {argv[0]} added; fragments removed")
    else:
        entries = fragments()
        print(section("(next)", entries).rstrip() if entries else f"no fragments in {CHANGES.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
