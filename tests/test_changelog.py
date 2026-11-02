"""scripts/changelog.py: fragments in PR order become the top section of WHATS_NEW.md and are removed; misnamed or
empty fragments, and a version already released, stop before anything is written."""
import pytest

from scripts import changelog


@pytest.fixture
def repo(tmp_path):
    changes, notes = tmp_path / "changes", tmp_path / "WHATS_NEW.md"
    changes.mkdir()
    (changes / "README.md").write_text("how to write a fragment\n", encoding="utf-8")
    notes.write_text("# What is new in basal 2.4.1\n\n- **Old entry**\n", encoding="utf-8")
    return changes, notes


def test_release_puts_fragments_in_pr_order_on_top(repo):
    changes, notes = repo
    (changes / "52.md").write_text("- **LoRA adapters**\n", encoding="utf-8")
    (changes / "9.md").write_text("- **Docs fix**\n  second line\n", encoding="utf-8")
    changelog.release("2.4.2", changes, notes)
    assert notes.read_text(encoding="utf-8") == (
        "# What is new in basal 2.4.2\n\n- **Docs fix**\n  second line\n- **LoRA adapters**\n\n"
        "# What is new in basal 2.4.1\n\n- **Old entry**\n")
    assert [f.name for f in changes.iterdir()] == ["README.md"]


@pytest.mark.parametrize("name, text", [("fix-cascade.md", "- x\n"), ("44.md", "\n  \n")])
def test_bad_fragment_stops_before_writing(repo, name, text):
    changes, notes = repo
    (changes / name).write_text(text, encoding="utf-8")
    before = notes.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        changelog.release("2.4.2", changes, notes)
    assert notes.read_text(encoding="utf-8") == before and (changes / name).exists()


def test_released_version_and_no_fragments_are_refused(repo):
    changes, notes = repo
    with pytest.raises(SystemExit, match="no fragments"):
        changelog.release("2.4.2", changes, notes)
    (changes / "44.md").write_text("- x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="already has a section"):
        changelog.release("2.4.1", changes, notes)
    assert (changes / "44.md").exists()


def test_repository_fragments_are_valid():
    changelog.fragments()
