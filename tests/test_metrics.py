import pytest

from marginalia.metrics import mean_reciprocal_rank, recall_at_k, reciprocal_rank

# Each test question is a pair: (what search returned, 1st to 5th — section names;
#                                which sections are the right answer).
# The same examples as in KONSPEKT, section 8.

RSI = (
    ["Elliott Wave", "RSI", "Свечи", "Глоссарий", "Объём"],
    {"RSI"},
)  # the right section is 2nd

HS_VOLUME = (
    ["Голова и плечи", "Elliott Wave", "Треугольники", "Свечи", "Гэпы"],
    {"Голова и плечи", "Важность объёма"},
)  # two right sections: one is 1st, the other was not found

MONEY = (
    ["Elliott Wave", "Свечи", "RSI", "Гэпы", "Треугольники"],
    {"Мани-менеджмент"},
)  # the right section was not found at all


def test_recall_at_k_examples():
    assert recall_at_k(*RSI, k=5) == 1.0  # found 1 of 1 right sections
    assert recall_at_k(*HS_VOLUME, k=5) == 0.5  # found 1 of 2
    assert recall_at_k(*MONEY, k=5) == 0.0  # found 0 of 1


def test_recall_only_looks_at_the_first_k():
    assert recall_at_k(*RSI, k=1) == 0.0  # RSI is 2nd, so it is not among the first 1
    assert recall_at_k(*RSI, k=2) == 1.0


def test_recall_counts_a_section_once_even_if_found_twice():
    # Two chunks of the RSI section were found; it is still one found section out of two right ones.
    assert recall_at_k(["RSI", "RSI", "Свечи"], {"RSI", "Глоссарий"}, k=3) == 0.5


def test_recall_with_k_larger_than_the_list():
    assert recall_at_k(["RSI"], {"RSI"}, k=10) == 1.0


def test_recall_without_right_sections_is_an_error():
    with pytest.raises(ValueError):
        recall_at_k(["RSI"], set(), k=5)  # unanswerable questions have no right sections


def test_reciprocal_rank():
    assert reciprocal_rank(*RSI) == 0.5  # first right one is 2nd → 1/2
    assert reciprocal_rank(*HS_VOLUME) == 1.0  # 1st → 1/1
    assert reciprocal_rank(*MONEY) == 0.0  # not found → 0
    assert reciprocal_rank(["Свечи", "Гэпы", "RSI"], {"RSI"}) == pytest.approx(1 / 3)  # 3rd → 1/3


def test_mean_reciprocal_rank_example():
    assert mean_reciprocal_rank([RSI, HS_VOLUME, MONEY]) == 0.5  # (0.5 + 1 + 0) / 3


def test_mean_reciprocal_rank_of_no_questions_is_an_error():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([])
