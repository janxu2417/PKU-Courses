"""简并 Rayleigh–Schrödinger 微扰论的精确与数值回归测试。"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from qmpert.degenerate import DegeneratePerturbationTheory, ResidualDegeneracyError


def _assert_exact_array(actual, expected):
    assert actual.shape == (len(expected),)
    for value, reference in zip(actual, expected):
        assert sp.simplify(value - reference) == 0


def test_lecture_example_exact_coefficients():
    """讲义式 (110)--(124)：低能态与高能二重简并块逐项精确核对。"""
    rt2 = sp.sqrt(2)
    e0 = np.array([0, 1, 1], dtype=object)
    V = np.array(
        [[0, rt2, 2 * rt2], [rt2, 0, 2], [2 * rt2, 2, 0]],
        dtype=object,
    )

    result = DegeneratePerturbationTheory(e0, V).solve()

    assert result.groups == ((0,), (1, 2))
    assert len(result.branches) == 3
    expected_corrections = [(0, -10), (-2, 1), (2, 9)]
    expected_c0 = [
        [1, 0, 0],
        [0, 1 / rt2, -1 / rt2],
        [0, 1 / rt2, 1 / rt2],
    ]
    expected_c1 = [
        [0, -rt2, -2 * rt2],
        [-1, 3 / (4 * rt2), 3 / (4 * rt2)],
        [3, -3 / (4 * rt2), 3 / (4 * rt2)],
    ]
    expected_e0 = [0, 1, 1]
    for branch, e0_ref, corrections, c0_ref, c1_ref in zip(
        result.branches, expected_e0, expected_corrections, expected_c0, expected_c1
    ):
        assert branch.e0 == e0_ref
        assert branch.dE1 == corrections[0]
        assert branch.dE2 == corrections[1]
        _assert_exact_array(branch.c0, c0_ref)
        _assert_exact_array(branch.c1, c1_ref)

    _assert_exact_array(result.U[:, 0], expected_c0[0])
    _assert_exact_array(result.U[:, 1], expected_c0[1])
    _assert_exact_array(result.U[:, 2], expected_c0[2])


def test_exact_branch_evaluation_stays_symbolic():
    rt2 = sp.sqrt(2)
    e0 = np.array([0, 1, 1], dtype=object)
    V = np.array(
        [[0, rt2, 2 * rt2], [rt2, 0, 2], [2 * rt2, 2, 0]],
        dtype=object,
    )
    branch = DegeneratePerturbationTheory(e0, V).solve().branches[1]
    lam = sp.symbols("lam")

    assert sp.simplify(branch.energy(lam) - (1 - 2 * lam + lam**2)) == 0
    vector = branch.state_vector(lam)
    _assert_exact_array(
        vector,
        [-lam, 1 / rt2 + 3 * lam / (4 * rt2), -1 / rt2 + 3 * lam / (4 * rt2)],
    )
    normalized = branch.normalized_state(lam)
    assert all(isinstance(value, sp.Basic) for value in normalized)
    assert not any(isinstance(value, (float, np.floating)) for value in normalized)


def test_random_degenerate_system_matches_small_lambda_diagonalization():
    rng = np.random.default_rng(2026)
    e0 = np.array([0.0, 1.0, 1.0, 3.0])
    A = rng.normal(size=(4, 4))
    V = (A + A.T) / 2
    result = DegeneratePerturbationTheory(e0, V).solve()
    lam = 1e-3
    H = np.diag(e0) + lam * V
    exact_energies, exact_states = np.linalg.eigh(H)

    assert result.groups == ((0,), (1, 2), (3,))
    assert len(result.branches) == 4
    for index, branch in enumerate(result.branches):
        assert abs(branch.energy(lam) - exact_energies[index]) < 1e-7
        vector = branch.state_vector(lam)
        residual = H @ vector - branch.energy(lam) * vector
        assert np.linalg.norm(residual) < 1e-5
        normalized = branch.normalized_state(lam)
        assert np.linalg.norm(normalized) == pytest.approx(1.0, abs=1e-14)
        # 小 λ 下各分支尚未交叉，符号/相位无关地对应同一精确本征态。
        overlap = abs(np.vdot(exact_states[:, index], normalized))
        assert overlap > 0.999


def test_nondegenerate_blocks_match_standard_rspt_formulas():
    e0 = np.array([0.0, 2.0, 5.0])
    V = np.array([[1.0, 2.0, -1.0], [2.0, -3.0, 0.5], [-1.0, 0.5, 2.0]])
    result = DegeneratePerturbationTheory(e0, V).solve()

    for n, branch in enumerate(result.branches):
        c1 = np.array(
            [0.0 if k == n else V[k, n] / (e0[n] - e0[k]) for k in range(3)]
        )
        dE2 = sum(abs(V[k, n]) ** 2 / (e0[n] - e0[k]) for k in range(3) if k != n)
        assert branch.e0 == e0[n]
        assert branch.dE1 == V[n, n]
        assert branch.dE2 == pytest.approx(dE2)
        assert np.allclose(branch.c1, c1)


def test_second_order_selects_basis_when_pvp_is_scalar():
    e0 = np.array([0, 0, 2], dtype=object)
    V = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=object)
    result = DegeneratePerturbationTheory(e0, V).solve()

    coupled, dark = result.branches[:2]
    assert (coupled.dE1, coupled.dE2) == (0, -1)
    assert (dark.dE1, dark.dE2) == (0, 0)
    _assert_exact_array(coupled.c0, [1 / sp.sqrt(2), 1 / sp.sqrt(2), 0])
    _assert_exact_array(coupled.c1, [0, 0, -1 / sp.sqrt(2)])


@pytest.mark.parametrize("dtype", [float, object])
def test_first_order_state_uses_third_order_projection_if_splitting_is_second_order(dtype):
    e0 = np.array([0, 0, 2, 3], dtype=dtype)
    V = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=dtype)
    branch = DegeneratePerturbationTheory(e0, V).solve().branches[0]

    assert float(branch.dE2) == pytest.approx(-0.5)
    assert np.asarray(branch.c1, dtype=float) == pytest.approx([0, -1, -0.5, 0])
    lam = 1e-3
    vector = np.linalg.eigh(np.diag(e0.astype(float)) + lam * V.astype(float))[1][:, 0]
    assert vector[1] / vector[0] == pytest.approx(-lam, abs=1e-8)


def test_energy_only_supports_degeneracy_through_second_order():
    e0 = [0, 0, 2, 2]
    V = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]
    solver = DegeneratePerturbationTheory(e0, V)
    result = solver.solve(energy_only=True)

    assert [b.dE2 for b in result.branches[:2]] == pytest.approx([-0.5, -0.5])
    assert all(b.c1 is None for b in result.branches)
    with pytest.raises(ResidualDegeneracyError, match="energy_only=True"):
        solver.solve()
    with pytest.raises(ResidualDegeneracyError):
        result.branches[0].state_vector(0.1)


def test_energy_origin_does_not_merge_distinct_degenerate_blocks():
    result = DegeneratePerturbationTheory([1e12, 1e12 + 1], [[0, 1], [1, 0]]).solve()
    assert result.groups == ((0,), (1,))
    assert result.branches[0].dE2 == -1


def test_small_energy_scale_preserves_distinct_blocks():
    result = DegeneratePerturbationTheory([0, 1e-19, 3e-19], np.eye(3) * 1e-21).solve()
    assert result.groups == ((0,), (1,), (2,))


def test_first_order_splitting_survives_a_large_perturbation_energy_offset():
    V = np.array([[0, 0.04, 0], [0.04, 0, 0.01], [0, 0.01, 0]])
    shifted = V + 1e15 * np.eye(3)
    result = DegeneratePerturbationTheory([0, 0, 1], shifted).solve()
    assert [b.dE2 for b in result.branches[:2]] == pytest.approx([-5e-5, -5e-5])


def test_rotated_scalar_projection_splits_at_second_order():
    e0 = [0.0, 0.0, 2.0]
    V = np.array([[0.3, 0, 1], [0, 0.3, 0.5], [1, 0.5, 0]])
    cosine, sine = np.cos(0.37), np.sin(0.37)
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    rotated = rotation.T @ V @ rotation
    branches = DegeneratePerturbationTheory(e0, rotated).solve().branches[:2]

    assert [b.dE2 for b in branches] == pytest.approx([-0.625, 0.0])
    assert max(np.linalg.norm(b.c1) for b in branches) < 10
    lam = 1e-3
    exact = np.linalg.eigvalsh(np.diag(e0) + lam * rotated)[:2]
    np.testing.assert_allclose(exact, np.linalg.eigvalsh(np.diag(e0) + lam * V)[:2],
                               atol=1e-14, rtol=0)
    np.testing.assert_allclose([b.energy(lam) for b in branches], exact, atol=1e-8, rtol=0)


def test_complex_hermitian_degenerate_block_has_unitary_good_basis():
    V = [[0, 1j, 1], [-1j, 0, 0], [1, 0, 0]]
    result = DegeneratePerturbationTheory([0, 0, 2], V).solve()

    assert [b.dE1 for b in result.branches] == pytest.approx([-1, 1, 0])
    assert [b.dE2 for b in result.branches] == pytest.approx([-0.25, -0.25, 0.5])
    np.testing.assert_allclose(result.U.conj().T @ result.U, np.eye(3), atol=1e-14, rtol=0)


def test_unsorted_spectrum_keeps_branch_energies_and_original_basis_indices():
    e0 = [1, 0, 1, 3]
    matrix = np.random.default_rng(2026).normal(size=(4, 4))
    V = (matrix + matrix.T) / 2
    result = DegeneratePerturbationTheory(e0, V).solve()

    assert result.groups == ((1,), (0, 2), (3,))
    assert [b.e0 for b in result.branches] == [0, 1, 1, 3]
    for branch in result.branches:
        outside = np.ones(4, dtype=bool)
        outside[list(result.groups[branch.block])] = False
        np.testing.assert_allclose(branch.c0[outside], 0, atol=1e-14, rtol=0)

    lam = 1e-3
    exact = np.linalg.eigvalsh(np.diag(e0) + lam * V)
    np.testing.assert_allclose([b.energy(lam) for b in result.branches], exact, atol=1e-7, rtol=0)


def test_fixed_width_integer_perturbation_has_no_wraparound():
    V = np.array([[100, 1], [1, 100]], dtype=np.int8)
    result = DegeneratePerturbationTheory([0, 0], V).solve()
    assert [b.dE1 for b in result.branches] == pytest.approx([99, 101])


def test_roundoff_does_not_couple_two_dark_states_to_external_block():
    V = np.zeros((4, 4))
    V[:3, 3] = V[3, :3] = [1, 2, 3]
    result = DegeneratePerturbationTheory([0, 0, 0, 1], V).solve()
    assert [b.dE2 for b in result.branches] == pytest.approx([-14, 0, 0, 14])
    assert np.linalg.norm(result.branches[2].c1) < 1e-13


def test_repeated_first_order_values_without_internal_mixing_are_allowed():
    e0 = np.array([0, 0, 2], dtype=object)
    V = np.diag([1, 1, 0]).astype(object)

    result = DegeneratePerturbationTheory(e0, V).solve()

    assert len(result.branches) == 3
    assert all(np.all(branch.c1 == 0) for branch in result.branches)
