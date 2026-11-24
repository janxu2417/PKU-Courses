"""
tests/test_qmpert.py —— 单元测试（pytest）

    pytest -q

``validate.py`` 汇总报告图表依赖的数值基准。本文件用于 CI 回归测试，覆盖部分基准，
外加 API 行为（异常、dtype、稀疏/稠密一致性）。全部用例应在数秒内跑完。
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from qmpert import (
    DegenerateSpectrumError,
    DeltaSymmetry,
    ParitySymmetry,
    PerturbationTheory,
    anharmonic_oscillator,
    banded_anharmonic,
    displaced_oscillator,
    random_model,
    symbolic_anharmonic,
    two_level,
)


# --------------------------------------------------------------------------- #
#  基础：前几阶系数
# --------------------------------------------------------------------------- #
def test_x4_first_orders():
    """E₀⁽¹⁾ = 3/4，E₀⁽²⁾ = −21/8（手工展开可验证）。"""
    m = anharmonic_oscillator(N=200)
    s = PerturbationTheory(m.e0, m.V).solve(0, 4)
    assert s.dE[1] == pytest.approx(0.75, abs=1e-13)
    assert s.dE[2] == pytest.approx(-21 / 8, abs=1e-12)
    assert s.measured_zero_orders() == set()


def test_x4_analytic_formula_all_states():
    """E_n⁽¹⁾ = (3/4)(2n²+2n+1)，E_n⁽²⁾ = −(34n³+51n²+59n+21)/8。"""
    m = anharmonic_oscillator(N=240)
    pt = PerturbationTheory(m.e0, m.V)
    for n in range(5):
        s = pt.solve(n, 3)
        assert s.dE[1] == pytest.approx(0.75 * (2 * n**2 + 2 * n + 1), rel=1e-12)
        a2 = -(34 * n**3 + 51 * n**2 + 59 * n + 21) / 8.0
        assert s.dE[2] == pytest.approx(a2, rel=1e-11)


def test_energy_series_evaluation():
    """energy(λ, m) 就是截断到 m 阶的部分和。"""
    m = anharmonic_oscillator(N=120)
    s = PerturbationTheory(m.e0, m.V).solve(0, 6)
    lam = 0.03
    manual = s.e0[0] + sum(s.dE[k] * lam**k for k in range(1, 7))
    assert s.energy(lam, 6) == pytest.approx(manual, rel=1e-14)


def test_state_vector_intermediate_normalization():
    """中间归一化 ⇒ ⟨0|n(λ)⟩ ≡ 1（每一阶的 n 分量都为 0）。"""
    m = anharmonic_oscillator(N=120)
    s = PerturbationTheory(m.e0, m.V).solve(0, 8)
    assert np.allclose(s.coeffs[1:, 0], 0.0)
    assert s.coeffs[0, 0] == 1.0


# --------------------------------------------------------------------------- #
#  与精确解对照
# --------------------------------------------------------------------------- #
def test_random_model_matches_diagonalization():
    """随机非简并模型：1–8 阶截断与严格对角化对照。"""
    m = random_model(N=10, seed=3)
    pt = PerturbationTheory(m.e0, m.V)
    s = pt.solve(2, 8)
    lam = 0.02
    exact = m.exact_spectrum(lam)[2]
    assert abs(s.energy(lam, 8) - exact) < 1e-12


def test_two_level_closed_form():
    """二能级：E₋ = Δ/2 − Δ/2·√(1+4v²/Δ²)，逐阶系数为二项式展开。"""
    m = two_level(delta=1.0)
    s = PerturbationTheory(m.e0, m.V).solve(0, 12)
    assert s.measured_zero_orders() == {1, 3, 5, 7, 9, 11}
    # E₋ = 1/2 − (1/2)√(1+4v²)：只有偶数阶非零，系数为 −(1/2)·C(1/2,k)·4^k
    from math import prod

    def binom_half(k: int) -> float:                     # 广义二项式 C(1/2, k)
        return prod(0.5 - j for j in range(k)) / prod(range(1, k + 1))

    for k in range(1, 13):
        if k % 2:
            assert s.dE[k] == pytest.approx(0.0, abs=1e-14)
        else:
            ref = -0.5 * binom_half(k // 2) * 4.0 ** (k // 2)
            assert s.dE[k] == pytest.approx(ref, rel=1e-10, abs=1e-14)
    v = 0.05                       # 4v² = 0.01，12 阶已足够收敛
    assert abs(s.energy(v, 12) - m.exact_energy(0, v)) < 1e-13


def test_complex_hermitian_perturbation():
    """复数 Hermitian 微扰：能量仍为实数，且与对角化一致。"""
    rng = np.random.default_rng(7)
    N = 8
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    V = (A + A.conj().T) / 2
    e0 = np.arange(N, dtype=float)
    pt = PerturbationTheory(e0, V)
    s = pt.solve(1, 6)
    assert np.max(np.abs(np.imag(s.dE))) < 1e-14
    H = np.diag(e0) + 0.02 * V
    # 六阶截断的残差 ~ λ⁷ ≈ 1e-12，故留 1e-10 余量
    assert abs(s.energy(0.02, 6) - np.linalg.eigvalsh(H)[1]) < 1e-10


# --------------------------------------------------------------------------- #
#  异常路径
# --------------------------------------------------------------------------- #
def test_degenerate_spectrum_raises():
    e0 = np.array([0.0, 0.0, 1.0])
    V = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    with pytest.raises(DegenerateSpectrumError):
        PerturbationTheory(e0, V).solve(0, 3)


def test_nondegenerate_solver_is_invariant_under_small_energy_scale():
    factor = 1e-19
    model = two_level(delta=1.0)
    original = PerturbationTheory(model.e0, model.V).solve(0, 4)
    scaled = PerturbationTheory(model.e0 * factor, model.V * factor).solve(0, 4)
    np.testing.assert_allclose(scaled.dE / factor, original.dE, atol=1e-14)
    np.testing.assert_allclose(scaled.coeffs, original.coeffs, atol=1e-14)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        PerturbationTheory(np.zeros(3), np.zeros((2, 2)))


def test_max_order_zero():
    m = anharmonic_oscillator(N=40)
    s = PerturbationTheory(m.e0, m.V).solve(0, 0)
    assert s.max_order == 0
    assert s.energy(0.1, 0) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
#  对称性
# --------------------------------------------------------------------------- #
def test_parity_predicts_odd_zeros():
    """奇宇称微扰 ⇒ 所有奇数阶能量修正为零，且预测 ⊆ 实测。"""
    m = displaced_oscillator(N=120)
    sym = ParitySymmetry((-1) ** np.arange(120), -1)
    pt = PerturbationTheory(m.e0, m.V, symmetry=sym)
    s = pt.solve(0, 10)
    assert s.zeros_predicted == frozenset({1, 3, 5, 7, 9})
    measured = s.measured_zero_orders()
    assert s.zeros_predicted <= measured
    assert all(type(order) is int for order in measured)


@pytest.mark.parametrize("sparse", [False, True])
def test_zero_orders_keep_small_complex_couplings_after_large_common_shift(sparse):
    from scipy.sparse import csr_matrix

    V = np.array([[1e8, 1e-4j], [-1e-4j, 1e8]])
    s = PerturbationTheory([0.0, 1.0], csr_matrix(V) if sparse else V).solve(0, 4)
    assert s.measured_zero_orders() == {3}
    assert s.measured_zero_orders(scale=1e8) == {2, 3, 4}


def test_zero_orders_detect_cancellation_when_low_orders_set_no_scale():
    V = [[0, 0.1, 0.3], [0.1, 0, 0], [0.3, 0, 0]]
    s = PerturbationTheory([0, 3, -27], V).solve(0, 3)
    assert abs(s.dE[2]) < 1e-16
    assert s.measured_zero_orders() == {1, 2, 3}


def test_displaced_oscillator_terminates():
    """位移谐振子的级数在二阶精确终止：E(λ) = 1/2 − F²/2 逐点成立。"""
    m = displaced_oscillator(N=100)
    s = PerturbationTheory(m.e0, m.V, symmetry=ParitySymmetry((-1) ** np.arange(100), -1)).solve(0, 6)
    for F in (0.1, 0.3, 0.7):
        assert abs(s.energy(F, 6) - m.exact_energy(0, F)) < 1e-13


def test_symmetry_mask_does_not_change_physics():
    """掩码只清掉求积噪声，不改变任何可观测结果。"""
    m = anharmonic_oscillator(N=160)
    sym = DeltaSymmetry([0, 2, -2, 4, -4])
    a = PerturbationTheory(m.e0, m.V, symmetry=sym).solve(0, 10)
    b = PerturbationTheory(m.e0, m.V).solve(0, 10)
    assert abs(a.energy(0.02, 10) - b.energy(0.02, 10)) < 1e-14


# --------------------------------------------------------------------------- #
#  稀疏后端
# --------------------------------------------------------------------------- #
def test_sparse_matches_dense_matrix():
    """两条独立构造路径给出同一 V。"""
    ms, md = banded_anharmonic(N=120), anharmonic_oscillator(N=120)
    assert np.abs(ms.V.toarray() - md.V).max() / np.abs(md.V).max() < 1e-14


def test_sparse_series_matches_dense():
    ms, md = banded_anharmonic(N=160), anharmonic_oscillator(N=160)
    a = PerturbationTheory(ms.e0, ms.V).solve(0, 20).dE
    b = PerturbationTheory(md.e0, md.V).solve(0, 20).dE
    assert np.max(np.abs(a - b)) / np.max(np.abs(b)) < 1e-12


def test_sparse_parity_mask_is_structural():
    """稀疏路径按 nnz 过滤，结果与稠密掩码逐元相同。"""
    m = banded_anharmonic(N=200)
    sym = ParitySymmetry((-1) ** np.arange(200), +1)
    eng = PerturbationTheory(m.e0, m.V, symmetry=sym)
    mask = sym.matrix_mask(200) & (m.V.toarray() != 0)
    assert np.array_equal(eng.v.astype(bool).toarray(), mask)


def test_sparse_large_basis_smoke():
    """N=2×10⁴ 的 40 阶层级（稠密需 3.2 GB）。"""
    m = banded_anharmonic(N=20_000)
    s = PerturbationTheory(m.e0, m.V).solve(0, 40)
    assert abs(s.dE[1] - 0.75) < 1e-13
    assert np.isfinite(s.energy(0.02, 20))


def test_sparse_exact_spectrum_agrees_with_dense():
    md = anharmonic_oscillator(N=300)
    ms = banded_anharmonic(N=300)
    assert abs(ms.exact_spectrum(0.05, k=1)[0] - md.exact_spectrum(0.05)[0]) < 1e-12


# --------------------------------------------------------------------------- #
#  精确算术
# --------------------------------------------------------------------------- #
def test_exact_coefficients_match_bender_wu():
    """sympy 后端给出精确有理系数，与文献逐位相同。"""
    e0, V = symbolic_anharmonic(N=20)
    s = PerturbationTheory(np.array(e0, dtype=object), np.array(V.tolist(), dtype=object)).solve(0, 5)
    assert s.energy_exact(0) == Fraction(1, 2)
    known = [Fraction(3, 4), Fraction(-21, 8), Fraction(333, 16),
             Fraction(-30885, 128), Fraction(916731, 256)]
    for m, ref in enumerate(known, start=1):
        assert Fraction(int(s.dE[m].p), int(s.dE[m].q)) == ref


def test_exact_matches_float_backend():
    e0, V = symbolic_anharmonic(N=14)
    s = PerturbationTheory(np.array(e0, dtype=object), np.array(V.tolist(), dtype=object)).solve(0, 5)
    f = anharmonic_oscillator(N=14)
    sf = PerturbationTheory(f.e0, f.V).solve(0, 5)
    for m in range(1, 6):
        assert float(s.dE[m]) == pytest.approx(sf.dE[m], rel=1e-13)


@pytest.mark.parametrize("float_type", [np.float32, np.float64, np.longdouble])
def test_object_perturbation_keeps_float_input_energies_exact(float_type):
    e0 = np.array([0.0, np.nextafter(float_type(0.7), float_type(1.0))], dtype=float_type)
    V = np.array([[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]], dtype=object)
    series = PerturbationTheory(e0, V).solve(0, 4)
    gap = Fraction(*e0[1].as_integer_ratio())

    assert all(isinstance(value, Fraction) for value in series.dE)
    assert series.dE[2] == -1 / gap
    assert series.dE[4] == 1 / gap**3
    assert series.coeffs[1, 1] == -1 / gap


def test_exact_energy_supports_sympy_irrational_coefficients():
    import sympy as sp

    V = np.array([[sp.sqrt(2), 0], [0, 0]], dtype=object)
    s = PerturbationTheory([0, 1], V).solve(0, 3)

    value = s.energy_exact(sp.Rational(1, 10))
    assert isinstance(value, sp.Expr)
    assert value == sp.sqrt(2) / 10


def test_exact_energy_respects_truncation_order():
    V = np.array([[0, Fraction(1)], [Fraction(1), 0]], dtype=object)
    s = PerturbationTheory([1, 2], V).solve(0, 4)
    lam = Fraction(1, 10)

    assert s.energy_exact(lam) == Fraction(9901, 10000)
    truncated = s.energy_exact(lam, order=2)
    assert isinstance(truncated, Fraction)
    assert truncated == Fraction(99, 100)
    assert s.energy_exact(lam, order=0) == Fraction(1)
    with pytest.raises(ValueError, match="order"):
        s.energy_exact(lam, order=5)


def test_exact_requires_object_dtype():
    m = anharmonic_oscillator(N=40)
    s = PerturbationTheory(m.e0, m.V).solve(0, 3)
    with pytest.raises(TypeError):
        s.energy_exact(0)


# --------------------------------------------------------------------------- #
#  重求和
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def x4_coeffs():
    m = anharmonic_oscillator(N=200)
    s = PerturbationTheory(m.e0, m.V).solve(0, 40)
    return m, np.concatenate(([s.unperturbed_energy], np.asarray(s.dE[1:], float))), s


def test_pade_reduces_to_taylor(x4_coeffs):
    from qmpert.resum import PadeTable

    _, c, s = x4_coeffs
    tab = PadeTable(c, max_order=12)
    # λ→0 时 [12/12] 必须退化为 12 阶 Taylor 多项式（差异 ~ c₁₃λ¹³）
    assert tab.at(1e-3, 12) == pytest.approx(s.energy(1e-3, 12), rel=1e-12)


def test_pade_beats_truncation(x4_coeffs):
    from qmpert.resum import PadeTable

    m, c, s = x4_coeffs
    lam = 0.2
    exact = m.exact_spectrum(lam)[0]
    errs = [abs(s.energy(lam, k) - exact) for k in range(1, 31)]
    assert min(errs) > 1e-3                     # 截断在此已失效
    assert abs(PadeTable(c, max_order=16).best(lam)[0] - exact) < 1e-6


def test_borel_pade_accuracy(x4_coeffs):
    from qmpert.resum import BorelPade

    m, c, _ = x4_coeffs
    lam = 0.2
    exact = m.exact_spectrum(lam)[0]
    assert abs(BorelPade(c, 16)(lam) - exact) < 1e-8


def test_borel_rejects_spurious_poles(x4_coeffs):
    """高阶 Borel–Padé 的分母在正实轴附近出现伪极点，必须被判定为不可用。"""
    from qmpert.resum import BorelPade

    _, c, _ = x4_coeffs
    bad = BorelPade(c, 20)
    assert not bad.is_safe(0.2)
    assert len(bad.dangerous_poles(0.2)) > 0


def test_pade_algebraic_identity(x4_coeffs):
    """Q·Σc − P 在匹配阶内应为零（高精度下）。"""
    from qmpert.resum import PadeTable

    _, c, _ = x4_coeffs
    tab = PadeTable(c, max_order=12)
    assert tab.approx[12].residual_mp < 1e-60
