"""Tests for trie deletion and pruning."""

from src.structures.trie import Trie


def test_delete_removes_key_only():
    t = Trie()
    for nm in ["car", "card", "care", "cat"]:
        t.insert(nm)
    assert t.delete("car")
    assert "car" not in t
    # prefixes/related keys survive
    assert "card" in t and "care" in t and "cat" in t
    assert sorted(t.starts_with("car")) == ["card", "care"]
    assert len(t) == 3


def test_delete_absent_returns_false():
    t = Trie()
    t.insert("abc")
    assert not t.delete("ab")          # not a stored word, only a prefix
    assert not t.delete("xyz")
    assert len(t) == 1


def test_delete_prunes_dangling_branch():
    t = Trie()
    t.insert("abcdef")
    t.insert("ab")
    assert t.delete("abcdef")
    # the unique deep branch is pruned, but the shared prefix word remains
    assert "ab" in t
    assert t.starts_with("abc") == []
    assert len(t) == 1
