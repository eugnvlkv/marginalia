from marginalia.index import get_section, ingest_book, search

# The fixtures embedder, collection and book_collection come from conftest.py.


def test_search_finds_the_closest_chunk(book_collection, embedder):
    hits = search(book_collection, embedder, "risk", k=1)

    assert [h.heading_path for h in hits] == [("Risk",)]
    assert hits[0].text == "Never risk more than you can afford to risk."


def test_hits_are_sorted_by_score_and_limited_to_k(book_collection, embedder):
    hits = search(book_collection, embedder, "trend", k=2)

    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
    assert hits[0].heading_path == ("Trends",)


def test_empty_heading_path_survives_the_round_trip(book_collection, embedder):
    # The preface has heading_path () — Chroma cannot store empty lists, we store a string.
    hits = search(book_collection, embedder, "trend", k=4)

    assert () in [h.heading_path for h in hits]


def test_reingest_replaces_old_chunks(book_collection, embedder):
    ingest_book(book_collection, "b", "# Risk\n\nOnly risk is left.\n", embedder)

    assert book_collection.count() == 1


def test_search_can_filter_by_book(collection, embedder):
    ingest_book(collection, "one", "# A\n\nA trend in book one.\n", embedder)
    ingest_book(collection, "two", "# A\n\nA trend in book two.\n", embedder)

    hits = search(collection, embedder, "trend", k=5, book_id="two")

    assert [h.book_id for h in hits] == ["two"]


LONG_SECTION = "\n\n".join(f"part {i} " + " ".join(["trend"] * 120) for i in range(3))


def test_get_section_glues_the_whole_section_back_in_order(collection, embedder):
    # Three ~730-char paragraphs do not fit into one 1500-char chunk: the section becomes 2 chunks.
    ingest_book(collection, "b", f"# Long\n\n{LONG_SECTION}\n\n# Other\n\nrisk here\n", embedder)

    section = get_section(collection, "b", ("Long",))

    assert section.chunk_ids == ("b:0", "b:1")
    assert section.text == LONG_SECTION


def test_get_section_returns_none_for_unknown_path(book_collection):
    assert get_section(book_collection, "b", ("No such section",)) is None
    assert get_section(book_collection, "other-book", ("Risk",)) is None
