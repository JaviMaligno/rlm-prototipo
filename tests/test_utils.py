from rlm.utils import estimate_tokens, load_documents


def test_estimate_tokens():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10


def test_load_documents(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello", encoding="utf-8")
    b.write_text("world", encoding="utf-8")

    doc = load_documents([str(a), str(b)])
    assert "FILE:" in doc.text
    assert doc.char_len == len(doc.text)
    assert doc.token_estimate == max(1, len(doc.text) // 4)
