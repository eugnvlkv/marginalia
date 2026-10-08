from marginalia.chunker import chunk_markdown
from marginalia.models import Chunk

BOOK = """Preface text before any heading.

# Chapter 4: Trends

Intro to trends.

## Trendlines

A trendline connects lows.

## Channels

A channel is two parallel lines.

# Chapter 5: Reversals

## Head and Shoulders

The most famous pattern.
"""

# What chunk_markdown(BOOK, "murphy") must return.
# "Chapter 5: Reversals" has no text of its own, so it gets no chunk,
# but its title stays in the path of "Head and Shoulders".
EXPECTED = [
    Chunk(
        id="murphy:0",
        book_id="murphy",
        heading_path=(),
        text="Preface text before any heading.",
        position=0,
    ),
    Chunk(
        id="murphy:1",
        book_id="murphy",
        heading_path=("Chapter 4: Trends",),
        text="Intro to trends.",
        position=1,
    ),
    Chunk(
        id="murphy:2",
        book_id="murphy",
        heading_path=("Chapter 4: Trends", "Trendlines"),
        text="A trendline connects lows.",
        position=2,
    ),
    Chunk(
        id="murphy:3",
        book_id="murphy",
        heading_path=("Chapter 4: Trends", "Channels"),
        text="A channel is two parallel lines.",
        position=3,
    ),
    Chunk(
        id="murphy:4",
        book_id="murphy",
        heading_path=("Chapter 5: Reversals", "Head and Shoulders"),
        text="The most famous pattern.",
        position=4,
    ),
]


def test_book_gives_expected_chunks():
    assert chunk_markdown(BOOK, "murphy") == EXPECTED


def test_splits_book_by_headings():
    chunks = chunk_markdown(BOOK, "murphy")

    assert [c.heading_path for c in chunks] == [
        (),
        ("Chapter 4: Trends",),
        ("Chapter 4: Trends", "Trendlines"),
        ("Chapter 4: Trends", "Channels"),
        ("Chapter 5: Reversals", "Head and Shoulders"),
    ]
    assert [c.text for c in chunks] == [
        "Preface text before any heading.",
        "Intro to trends.",
        "A trendline connects lows.",
        "A channel is two parallel lines.",
        "The most famous pattern.",
    ]


def test_ids_and_positions_follow_chunk_order():
    chunks = chunk_markdown(BOOK, "murphy")

    assert [c.position for c in chunks] == [0, 1, 2, 3, 4]
    assert [c.id for c in chunks] == [
        "murphy:0",
        "murphy:1",
        "murphy:2",
        "murphy:3",
        "murphy:4",
    ]
    assert all(c.book_id == "murphy" for c in chunks)


def test_nested_levels_empty_section_and_fake_heading():
    md = "# A\n## B\ntext b\n### C\ntext c\n## D\n#1 rule: cut losses\n"

    chunks = chunk_markdown(md, "b")

    assert [(c.heading_path, c.text) for c in chunks] == [
        (("A", "B"), "text b"),
        (("A", "B", "C"), "text c"),
        (("A", "D"), "#1 rule: cut losses"),
    ]


def test_text_keeps_inner_blank_lines_and_trims_edges():
    md = "# A\n\n\nFirst paragraph.\n\nSecond paragraph.\n\n\n"

    chunks = chunk_markdown(md, "b")

    assert chunks[0].text == "First paragraph.\n\nSecond paragraph."


def test_hash_without_space_or_more_than_six_is_text():
    md = "# A\n#hashtag\n####### seven\n"

    chunks = chunk_markdown(md, "b")

    assert chunks[0].text == "#hashtag\n####### seven"


def test_heading_title_is_trimmed():
    chunks = chunk_markdown("## Trendlines  \ntext\n", "b")

    assert chunks[0].heading_path == ("Trendlines",)


def test_long_section_is_split_into_chunks_with_the_same_path():
    md = "# A\n\naaaa\n\nbbbb\n\ncccccc\n\ndd\n\n# B\n\ntext b\n"

    chunks = chunk_markdown(md, "b", max_chars=12)

    assert [(c.heading_path, c.text, c.position) for c in chunks] == [
        (("A",), "aaaa\n\nbbbb", 0),
        (("A",), "cccccc\n\ndd", 1),
        (("B",), "text b", 2),
    ]


def test_no_text_gives_no_chunks():
    assert chunk_markdown("", "b") == []
    assert chunk_markdown("# A\n## B\n", "b") == []
