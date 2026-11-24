"""重求和的高阶边界与奇异阶回归测试。"""

from math import factorial

import numpy as np
import pytest

from qmpert import PerturbationTheory, two_level
from qmpert.resum import PadeTable, _borel_coeffs


def test_borel_coefficients_drop_subnormal_unit_coefficients_above_170():
    coeffs = _borel_coeffs(np.ones(200))
    assert coeffs.shape == (200,)
    assert coeffs.dtype == np.float64
    for k in (0, 5, 170):
        assert coeffs[k] == 1.0 / factorial(k)
    assert np.all(coeffs[171:] == 0)


def test_borel_coefficients_preserve_representable_normal_quotients_above_170():
    coeffs = np.zeros(350)
    coeffs[199] = -1e300
    coeffs[300:302] = np.finfo(float).max
    transformed = _borel_coeffs(coeffs)

    for k in (199, 300):
        numerator, denominator = float(coeffs[k]).as_integer_ratio()
        assert transformed[k] == numerator / (denominator * factorial(k))
        assert transformed[k] != 0
    assert transformed[301] == 0


@pytest.mark.parametrize("high_precision", [True, False])
def test_pade_table_skips_singular_odd_orders(high_precision):
    model = two_level()
    coeffs = PerturbationTheory(model.e0, model.V).solve(0, 40).dE
    table = PadeTable(coeffs, 20, high_precision=high_precision)

    assert set(table.approx) == set(range(2, 21, 2))
    value, order, _ = table.best(0.3)
    assert order % 2 == 0
    assert abs(value - model.exact_energy(0, 0.3)) < 1e-10


@pytest.mark.parametrize("coeffs", [[0, 0, -1], [0, 0, -1, 0, 1]])
def test_pade_best_requires_two_usable_orders(coeffs):
    table = PadeTable(coeffs)
    assert len(table.approx) < 2
    with pytest.raises(ValueError, match="两个可用"):
        table.best(0.3)
