"""逐阶范数归一化系数的回归测试。"""

from fractions import Fraction

import numpy as np
import pytest

from qmpert.core import PerturbationTheory


def _two_level(dtype):
    e0 = np.array([-Fraction(1, 2), Fraction(1, 2)], dtype=dtype)
    V = np.array(
        [[0, Fraction(1, 2)], [Fraction(1, 2), 0]], dtype=dtype
    )
    return PerturbationTheory(e0, V).solve(0, 6)


def test_exact_two_level_normalized_coeffs():
    series = _two_level(object)
    u = series.normalized_coeffs()

    assert u.shape == series.coeffs.shape
    assert u.dtype == object
    assert u[2, series.state] == Fraction(-1, 8)
    for m in range(series.max_order + 1):
        norm_order = sum(
            u[k, i].conjugate() * u[m - k, i]
            for k in range(m + 1)
            for i in range(u.shape[1])
        )
        assert norm_order == Fraction(m == 0)


def test_sympy_normalized_coeffs_remain_exact():
    from qmpert.models import symbolic_anharmonic

    e0, V = symbolic_anharmonic(N=12)
    series = PerturbationTheory(
        np.array(e0, dtype=object), np.array(V.tolist(), dtype=object)
    ).solve(0, 4)
    u = series.normalized_coeffs()

    assert not any(isinstance(x, (float, complex)) for x in u.flat)
    for m in range(series.max_order + 1):
        norm_order = sum(
            u[k, i].conjugate() * u[m - k, i]
            for k in range(m + 1)
            for i in range(u.shape[1])
        )
        assert norm_order == int(m == 0)


def test_complex_hermitian_normalized_coeffs():
    rng = np.random.default_rng(19)
    n_basis = 6
    A = rng.normal(size=(n_basis, n_basis)) + 1j * rng.normal(
        size=(n_basis, n_basis)
    )
    V = (A + A.conj().T) / (4 * n_basis)
    e0 = 1.3 * np.arange(n_basis)
    series = PerturbationTheory(e0, V).solve(2, 8)
    u = series.normalized_coeffs()

    for m in range(series.max_order + 1):
        norm_order = sum(
            np.vdot(u[k], u[m - k]) for k in range(m + 1)
        )
        assert norm_order == pytest.approx(float(m == 0), abs=2e-11)


def test_unit_series_matches_normalized_state_at_small_lambda():
    e0 = np.array([-0.5, 0.5])
    V = np.array([[0.0, 0.5], [0.5, 0.0]])
    series = PerturbationTheory(e0, V).solve(0, 4)
    u = series.normalized_coeffs()

    errors = []
    for lam in (0.08, 0.04):
        truncated = sum(lam**m * u[m] for m in range(5))
        reference = series.normalized_state(lam, order=4)
        errors.append(np.linalg.norm(truncated - reference))

    assert errors[1] < 1e-8
    assert errors[1] < errors[0] * 0.5**4


def test_numeric_integer_spectrum_is_a_valid_input():
    series = PerturbationTheory([0, 3], [[0, 1], [1, 0]]).solve(0, 4)
    assert series.dE[2] == pytest.approx(-1 / 3)
    assert series.dE[4] == pytest.approx(1 / 27)


@pytest.mark.parametrize("e0", [[0, 3], np.array([0, 3], dtype=object)])
def test_integer_spectrum_with_object_perturbation_retains_exactness(e0):
    V = np.array([[0, 1], [1, 0]], dtype=object)
    series = PerturbationTheory(e0, V).solve(0, 4)

    assert series.dE[2] == Fraction(-1, 3)
    assert series.dE[4] == Fraction(1, 27)
    assert series.energy(Fraction(1, 2), 2) == Fraction(-1, 12)
    assert series.state_vector(Fraction(1, 2), 1)[1] == Fraction(-1, 6)
    assert series.normalized_coeffs()[2, 0] == Fraction(-1, 18)


def test_high_order_growth_does_not_erase_low_order_coefficients():
    from qmpert.models import anharmonic_oscillator

    model = anharmonic_oscillator(N=100)
    series = PerturbationTheory(model.e0, model.V).solve(0, 40)
    assert series.measured_zero_orders() == set()


def test_energy_origin_does_not_change_nondegenerate_classification():
    V = [[0.0, 1.0], [1.0, 0.0]]
    series = PerturbationTheory([1e12, 1e12 + 1], V).solve(0, 2)
    assert series.dE[2] == -1
