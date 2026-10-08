from marginalia.chunker import split_text


def test_example_from_the_lesson():
    text = "aaaa\n\nbbbb\n\ncccccc\n\ndd"

    print(split_text(text, 12))

    assert split_text(text, 12) == ["aaaa\n\nbbbb", "cccccc\n\ndd"]


def test_short_text_stays_one_piece():
    assert split_text("one\n\ntwo", 2000) == ["one\n\ntwo"]


def test_piece_of_exactly_max_chars_fits():
    # 4 + 2 + 6 = 12: a piece exactly at the limit is allowed.
    assert split_text("aaaa\n\ncccccc", 12) == ["aaaa\n\ncccccc"]


def test_paragraph_longer_than_max_is_kept_whole():
    long = "x" * 20

    assert split_text(f"short\n\n{long}\n\nend", 10) == ["short", long, "end"]


def test_empty_text_gives_no_pieces():
    assert split_text("", 12) == []
