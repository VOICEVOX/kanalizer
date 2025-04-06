import pytest

import kanalizer


def test_kanalizer():
    word = "kanalizer"
    assert kanalizer.convert(word) == "カナライザー"


def test_invalid_max_length():
    word = "kanalizer"
    with pytest.raises(ValueError):
        kanalizer.convert(word, max_length=0)


def test_empty_word():
    word = ""
    with pytest.raises(ValueError):
        kanalizer.convert(word)


def test_invalid_chars():
    word = "あ"
    with pytest.raises(ValueError):
        kanalizer.convert(word)
