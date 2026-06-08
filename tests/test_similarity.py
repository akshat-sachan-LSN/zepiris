import pytest

from zepiris.services.similarity import cosine


def test_identical_vectors_score_one() -> None:
    assert cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero() -> None:
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_score_minus_one() -> None:
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_zero_vector_returns_zero() -> None:
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
