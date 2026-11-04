import pytest
from pydantic import ValidationError

from marginalia.golden import GoldenSet, check_golden, covers, find_sections, load_golden

# book_collection (conftest.py) holds book "b": a preface, "Trends", "Risk", "Candles".


def golden(*questions):
    return GoldenSet.model_validate({"questions": list(questions)})


def q(id, kind="factual", relevant=(("b", "Risk"),), expected="something"):
    return {
        "id": id,
        "kind": kind,
        "question": "q?",
        "relevant": [{"book": book, "path": path} for book, path in relevant],
        "expected": expected,
    }


def test_load_golden_reads_toml(tmp_path):
    file = tmp_path / "golden.toml"
    file.write_text(
        '[[questions]]\nid = "a"\nkind = "unanswerable"\nquestion = "q?"\n'
        '\n[[questions]]\nid = "b"\nkind = "factual"\nquestion = "q?"\n'
        'relevant = [{ book = "b", path = "Risk" }]\nexpected = "x"\n',
        encoding="utf-8",
    )

    loaded = load_golden(file)

    assert [x.id for x in loaded.questions] == ["a", "b"]
    assert loaded.questions[1].relevant[0].path == "Risk"


def test_a_typo_in_a_field_name_is_an_error():
    with pytest.raises(ValidationError):
        golden({**q("a"), "relevent": []})


def test_a_valid_golden_set_has_no_problems(book_collection):
    good = golden(
        q("a"),
        q("b", kind="multi", relevant=(("b", "Risk"), ("b", "Trends"))),
        q("c", kind="unanswerable", relevant=(), expected=""),
        q("d", kind="partial"),
    )

    assert check_golden(good, book_collection) == []


def test_check_finds_the_problems(book_collection):
    bad = golden(
        q("dup"),
        q("dup"),
        q("typo", relevant=(("b", "Riks"),)),
        q("wrong-book", relevant=(("other", "Risk"),)),
        q("refusal-with-sections", kind="unanswerable"),
        q("factual-without-sections", relevant=()),
        q("multi-with-one", kind="multi"),
        q("no-expected", expected=" "),
        q("partial-without-sections", kind="partial", relevant=()),
    )

    problems = check_golden(bad, book_collection)

    assert "dup: id is used 2 times" in problems
    assert "typo: no section 'Riks' in book 'b'" in problems
    assert "wrong-book: no section 'Risk' in book 'other'" in problems
    assert "refusal-with-sections: unanswerable but has relevant sections" in problems
    assert "factual-without-sections: factual needs at least one relevant section" in problems
    assert "multi-with-one: multi needs at least two relevant sections" in problems
    assert "no-expected: expected is empty — the judge has nothing to compare with" in problems
    assert "partial-without-sections: partial needs at least one relevant section" in problems


def test_a_section_covers_its_subsections():
    assert covers("A › B", "A › B")
    assert covers("A › B", "A › B › C")
    assert not covers("A › B", "A › BC")  # a prefix of a word is not a subsection
    assert not covers("A › B", "A")


def test_find_sections_matches_any_case(book_collection):
    assert find_sections(book_collection, "risk") == [("b", "Risk")]


def test_section_text_includes_subsections_in_book_order(collection, embedder):
    from marginalia.golden import section_text
    from marginalia.index import ingest_book

    ingest_book(collection, "b", "# A\n\nintro trend\n\n## B\n\nsub risk\n\n# C\n\nother candle\n", embedder)

    text = section_text(collection, "b", "A")

    assert text == "── A\n\nintro trend\n\n── A › B\n\nsub risk"


def test_promote_moves_only_reviewed_blocks_with_their_comments(tmp_path):
    from marginalia.golden import promote

    source = tmp_path / "candidates.toml"
    target = tmp_path / "golden.toml"
    source.write_text(
        "# header\n\n"
        '[[questions]]\nid = "a"\nkind = "unanswerable"\nquestion = "q?"\nreviewed = true\n# note: keep me\n\n'
        '[[questions]]\nid = "b"\nkind = "unanswerable"\nquestion = "q?"\n',
        encoding="utf-8",
    )
    target.write_text("# golden\n", encoding="utf-8")

    moved = promote(source, target)

    assert moved == ["a"]
    assert '# note: keep me' in target.read_text(encoding="utf-8")
    assert [q.id for q in load_golden(target).questions] == ["a"]
    assert [q.id for q in load_golden(source).questions] == ["b"]
    assert source.read_text(encoding="utf-8").startswith("# header")
