"""数值微扰矩阵的厄米性告警回归测试。"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from qmpert import DeltaSymmetry, DegeneratePerturbationTheory, PerturbationTheory


@pytest.mark.parametrize("solver", [PerturbationTheory, DegeneratePerturbationTheory])
@pytest.mark.parametrize(
    "V",
    [
        np.array([[0, 100], [-100, 0]], dtype=np.int8),
        np.array([[0, 1j], [1j, 0]]),
        np.array([[0, 1e-200], [0, 0]]),
        np.array([[0, 1e200], [0, 0]]),
    ],
)
def test_non_hermitian_numeric_matrix_warns_at_call_site(solver, V):
    original = V.copy()
    with pytest.warns(RuntimeWarning, match="厄米") as captured:
        solver([0, 1], V)

    assert Path(captured[0].filename).resolve() == Path(__file__).resolve()
    np.testing.assert_array_equal(V, original)


@pytest.mark.parametrize("solver", [PerturbationTheory, DegeneratePerturbationTheory])
@pytest.mark.parametrize(
    "V",
    [
        np.array([[1, 1j], [-1j, -1]]),
        np.array([[0, 1 + 1e-12j], [1, 0]]),
        np.array([[0, 1], [0, 0]], dtype=object),
    ],
)
def test_hermitian_noise_and_symbolic_backend_do_not_warn(solver, V):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        solver([0, 1], V)


def test_sparse_hermiticity_check_does_not_densify(monkeypatch):
    nonhermitian = csr_matrix(([1j], ([0], [1])), shape=(2, 2))
    hermitian = nonhermitian + nonhermitian.conj().T

    def fail_toarray(*args, **kwargs):
        pytest.fail("稀疏厄米性校验不应转为稠密矩阵")

    monkeypatch.setattr(csr_matrix, "toarray", fail_toarray)
    with pytest.warns(RuntimeWarning, match="厄米"):
        PerturbationTheory([0, 1], nonhermitian)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        PerturbationTheory([0, 1], hermitian)
        PerturbationTheory([0, 1], csr_matrix((2, 2)))


def test_hermiticity_check_uses_perturbation_after_symmetry_mask():
    V = np.array([[0, 1], [1, 0]])
    with pytest.warns(RuntimeWarning, match="厄米"):
        PerturbationTheory([0, 1], V, symmetry=DeltaSymmetry({1}))
