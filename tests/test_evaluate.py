from marginalia.evaluate import Scored, evaluate, label_hits, outermost, summarize
from marginalia.golden import GoldenQuestion, GoldenSet, Relevant
from marginalia.index import Hit


def hit(path, book="b"):
    """A found chunk from `book` that lies in section `path` ("Chapter › Section")."""
    return Hit(chunk_id="c", book_id=book, heading_path=tuple(path.split(" › ")), text="", score=0.0)


def right(path, book="b"):
    """A right section from the golden set."""
    return Relevant(book=book, path=path)


def test_a_hit_in_a_right_section_gets_that_sections_key():
    assert label_hits([hit("Risk")], [right("Risk")]) == ["b: Risk"]


def test_a_hit_in_a_subsection_counts_for_the_right_section():
    assert label_hits([hit("Risk › Stops")], [right("Risk")]) == ["b: Risk"]


def test_a_hit_outside_the_right_sections_keeps_its_own_key():
    assert label_hits([hit("Trends")], [right("Risk")]) == ["b: Trends"]


def test_a_similar_name_is_not_the_same_section():
    # "Risk" is part of the text "Risk management", but "Risk management" is another section, not a subsection of "Risk".
    assert label_hits([hit("Risk management")], [right("Risk")]) == ["b: Risk management"]
    # "Risk › Stops" ends with "Stops", but it lies in "Risk", not in the top-level section "Stops".
    assert label_hits([hit("Risk › Stops")], [right("Stops")]) == ["b: Risk › Stops"]


def test_the_book_must_match_too():
    assert label_hits([hit("Risk", book="other")], [right("Risk")]) == ["other: Risk"]


def test_labels_keep_the_search_order():
    hits = [hit("Trends"), hit("Risk"), hit("Candles")]
    assert label_hits(hits, [right("Risk"), right("Candles")]) == ["b: Trends", "b: Risk", "b: Candles"]


def test_outermost_drops_sections_nested_in_another_right_section():
    assert outermost([right("A"), right("A › B"), right("C")]) == [right("A"), right("C")]


def test_summarize_averages_per_kind_and_overall():
    results = [
        Scored(GoldenQuestion(id="a", kind="factual", question="q?"), ["b: Risk", "b: Trends"], {"b: Risk"}),
        Scored(GoldenQuestion(id="m", kind="multi", question="q?"), ["b: Trends", "b: Risk"], {"b: Risk", "b: Candles"}),
    ]

    summary = summarize(results, k=2)

    assert summary["factual"] == (1, 1.0, 1.0)  # found 1 of 1, first right one 1st
    assert summary["multi"] == (1, 0.5, 0.5)  # found 1 of 2, first right one 2nd
    assert summary["all"] == (2, 0.75, 0.75)


def test_evaluate_runs_the_real_search_and_leaves_out_unanswerable(book_collection, embedder):
    golden = GoldenSet.model_validate(
        {
            "questions": [
                {"id": "risk", "kind": "factual", "question": "risk or not", "relevant": [{"book": "b", "path": "Risk"}]},
                {"id": "btc-now", "kind": "unanswerable", "question": "trend of BTC now"},
            ]
        }
    )

    results = evaluate(golden, book_collection, embedder, k=1)

    assert [r.question.id for r in results] == ["risk"]
    assert results[0].labels == ["b: Risk"]  # the Risk chunk is the closest to "risk or not"
    assert summarize(results, k=1)["all"] == (1, 1.0, 1.0)
