"""二阶能量、一阶态矢的简并 Rayleigh–Schrödinger 微扰论。

输入是 H₀ 本征基下的本征值与 V 矩阵，H(λ)=H₀+λV。每个简并块
先对角化 PVP 以选好基；结果中的态矢系数始终转回原来的 H₀ 本征基。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .core import _warn_if_nonhermitian

__all__ = [
    "DegenerateBranch",
    "DegeneratePerturbationResult",
    "DegeneratePerturbationTheory",
    "ResidualDegeneracyError",
]


class ResidualDegeneracyError(ValueError):
    """需要更高阶有效矩阵才能确定态矢。"""


@dataclass
class DegenerateBranch:
    """一个好基分支的微扰结果，采用中间归一化 c1[α]=0。"""

    good_index: int
    block: int
    e0: object
    dE1: object
    dE2: object
    c0: np.ndarray
    c1: np.ndarray | None

    def energy(self, lam):
        """返回 E₀+λE₁+λ²E₂。"""
        return self.e0 + lam * self.dE1 + lam**2 * self.dE2

    def state_vector(self, lam) -> np.ndarray:
        """返回截断到一阶的原基态矢 c₀+λc₁（未归一化）。"""
        if self.c1 is None:
            raise ResidualDegeneracyError("energy_only=True 的结果只含能量和零阶好基。")
        return self.c0 + lam * self.c1

    def normalized_state(self, lam) -> np.ndarray:
        """归一化截断态矢；object 后端保持 SymPy 精确运算。"""
        vector = self.state_vector(lam)
        if vector.dtype == object:
            import sympy as sp

            norm2 = sp.simplify(sum(sp.conjugate(x) * x for x in vector))
            norm = sp.sqrt(norm2)
            return np.array([sp.simplify(x / norm) for x in vector], dtype=object)
        return vector / np.linalg.norm(vector)


@dataclass
class DegeneratePerturbationResult:
    """所有分支、好基变换 U 及原基指标分组。"""

    branches: list[DegenerateBranch]
    U: np.ndarray
    groups: tuple[tuple[int, ...], ...]

    @property
    def good_basis(self) -> np.ndarray:
        """``U`` 的语义化别名。"""
        return self.U


class DegeneratePerturbationTheory:
    """自动分组的二阶简并微扰求解器。

    Parameters
    ----------
    h0_energies : (N,) array
        H₀ 的本征值。
    v_matrix : (N, N) array
        同一 H₀ 本征基下的厄米微扰矩阵。数值后端会检查厄米性。
    degeneracy_tol : float
        数值后端的能级分组与剩余简并判据；object/SymPy 后端只用精确相等。
    """

    def __init__(
        self,
        h0_energies: Sequence,
        v_matrix: Sequence,
        degeneracy_tol: float = 1e-10,
    ) -> None:
        self.e0 = self._asarray(h0_energies)
        self.V = self._asarray(v_matrix)
        if self.e0.ndim != 1 or self.e0.size == 0:
            raise ValueError("h0_energies 必须是非空 1D 数组")
        n = self.e0.size
        if self.V.shape != (n, n):
            raise ValueError(f"v_matrix 形状应为 {(n, n)}，收到 {self.V.shape}")
        self.exact = self.e0.dtype == object or self.V.dtype == object
        if self.exact:
            import sympy as sp

            self.e0 = np.array([sp.sympify(x) for x in self.e0], dtype=object)
            self.V = np.array(sp.Matrix(self.V.tolist()).tolist(), dtype=object)
        else:
            if np.any(np.imag(self.e0) != 0) or not np.all(np.isfinite(self.e0)):
                raise ValueError("h0_energies 必须是有限实数")
            self.e0 = np.real(self.e0).astype(np.result_type(self.e0.real.dtype, np.float64))
            self.V = self.V.astype(np.result_type(self.V.dtype, np.float64))
        self.degeneracy_tol = float(degeneracy_tol)
        if not np.isfinite(self.degeneracy_tol) or self.degeneracy_tol < 0:
            raise ValueError("degeneracy_tol 必须是有限非负数")
        self.n_basis = n
        if not self.exact:
            _warn_if_nonhermitian(self.V)

    @staticmethod
    def _asarray(values) -> np.ndarray:
        # SymPy Matrix 先转嵌套列表，避免 NumPy 将其当作单个 object 标量。
        if not isinstance(values, np.ndarray) and hasattr(values, "tolist"):
            values = values.tolist()
        return np.asarray(values)

    def _partition(self, energies, scale_floor=0.0) -> tuple[tuple[int, ...], ...]:
        if self.exact:
            groups: list[list[int]] = []
            for index, energy in enumerate(energies):
                for group in groups:
                    if self._exact_zero(energy - energies[group[0]]):
                        group.append(index)
                        break
                else:
                    groups.append([index])
            return tuple(tuple(group) for group in groups)

        order = np.argsort(energies, kind="stable")
        groups = []
        tolerance = self.degeneracy_tol * max(scale_floor, float(np.ptp(energies)))
        for index in order:
            index = int(index)
            if not groups or abs(energies[index] - energies[groups[-1][0]]) > tolerance:
                groups.append([index])
            else:
                groups[-1].append(index)
        return tuple(tuple(group) for group in groups)

    @staticmethod
    def _canonicalize_numeric(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector)
        nonzero = np.flatnonzero(np.abs(vector) > 1e-14)
        if nonzero.size:
            pivot = vector[nonzero[0]]
            vector = vector * (np.conjugate(pivot) / abs(pivot))
            if abs(np.imag(vector[nonzero[0]])) < 1e-14:
                vector[nonzero[0]] = np.real(vector[nonzero[0]])
        return vector

    @staticmethod
    def _canonicalize_exact(vector):
        import sympy as sp

        vector = sp.Matrix(vector)
        vector = vector / vector.norm()
        for pivot in vector:
            if sp.simplify(pivot) != 0:
                phase = sp.conjugate(pivot) / sp.Abs(pivot)
                vector = phase * vector
                break
        return vector.applyfunc(sp.simplify)

    def _eigensystem(self, block: np.ndarray):
        if self.exact:
            import sympy as sp

            matrix = sp.Matrix(block.tolist())
            eigenvalues = []
            vectors = []
            for value, _multiplicity, basis in matrix.eigenvects():
                for vector in sp.GramSchmidt(basis):
                    eigenvalues.append(sp.simplify(value))
                    vectors.append(self._canonicalize_exact(vector))
            return eigenvalues, np.array(sp.Matrix.hstack(*vectors).tolist(), dtype=object)
        block = (block + block.conj().T) / 2
        eigenvalues, vectors = np.linalg.eigh(block)
        vectors = np.column_stack([self._canonicalize_numeric(v) for v in vectors.T])
        return eigenvalues, vectors

    def _build_good_basis(self, groups, perturbation):
        dtype = object if self.exact else np.result_type(self.e0, self.V, np.float64)
        U = np.zeros((self.n_basis, self.n_basis), dtype=dtype)
        block_data = []
        good_energies = []
        column = 0
        for indices in groups:
            projected = perturbation[np.ix_(indices, indices)]
            # 保留块内相对劈裂，用绝对一阶能量输出时再加回公共项。
            offset = 0 if self.exact else projected[0, 0].real
            projected -= offset * np.eye(len(indices), dtype=U.dtype)
            eigenvalues, vectors = self._eigensystem(projected)
            columns = tuple(range(column, column + len(indices)))
            U[np.ix_(indices, columns)] = vectors
            block_data.append((columns, self.e0[indices[0]], eigenvalues, offset))
            good_energies.extend([self.e0[indices[0]]] * len(indices))
            column += len(indices)
        return U, block_data, good_energies

    def _simplify(self, matrix) -> np.ndarray:
        if self.exact:
            import sympy as sp

            return np.vectorize(sp.simplify, otypes=[object])(matrix)
        return matrix

    @staticmethod
    def _exact_zero(value) -> bool:
        import sympy as sp

        return sp.simplify(value) == 0

    def _state_corrections(
        self, block, first_order, first_groups, second_order, W2, X, C, w2_scale=0.0
    ):
        """在 P 内解一阶混合。X=(E0-H0_Q)^(-1) V_QP，C=V_QQ。"""
        within = np.zeros_like(W2)
        local = set(range(len(first_order)))
        if not self.exact:
            roundoff = 8 * np.finfo(X.real.dtype).eps * np.max(np.abs(X), initial=0.0)
        for group in first_groups:
            other = sorted(local.difference(group))
            for alpha in group:
                for beta in other:
                    within[beta, alpha] = W2[beta, alpha] / (
                        first_order[alpha] - first_order[beta]
                    )

        for group in first_groups:
            if len(group) == 1:
                continue
            other = sorted(local.difference(group))
            Xg = X[:, group]
            e1 = first_order[group[0]]
            # 消去 Q；再消去 P 中具有不同一阶能量的态。
            K3 = Xg.conj().T @ ((C - e1 * np.eye(len(C), dtype=C.dtype)) @ Xg)
            K3 += W2[np.ix_(group, other)] @ within[np.ix_(other, group)]
            seconds = second_order[list(group)]
            scale = 0.0 if self.exact else max(float(np.max(np.abs(seconds))), w2_scale)
            split = self._partition(seconds, scale_floor=scale)
            for subgroup in split:
                if len(subgroup) > 1:
                    coupling = Xg[:, subgroup]
                    if self.exact:
                        coupled = any(not self._exact_zero(value) for value in coupling.flat)
                    else:
                        coupled = np.any(np.abs(coupling) > roundoff)
                    if coupled:
                        raise ResidualDegeneracyError(
                            f"简并块 {block} 到二阶仍有耦合分支简并；态矢需要更高阶选基，"
                            "可用 solve(energy_only=True) 求二阶能量。"
                        )
                for a in subgroup:
                    for b in sorted(set(range(len(group))).difference(subgroup)):
                        alpha, beta = group[a], group[b]
                        within[beta, alpha] = K3[b, a] / (second_order[alpha] - second_order[beta])
        return within

    def solve(self, *, energy_only: bool = False) -> DegeneratePerturbationResult:
        """求二阶能量；默认同时求一阶态矢，``energy_only`` 下各分支的 ``c1`` 为 None。"""
        groups = self._partition(self.e0)
        offset = 0 if self.exact else self.V[0, 0].real
        perturbation = self.V - offset * np.eye(self.n_basis, dtype=self.V.dtype)
        first_basis, block_data, good_energies = self._build_good_basis(groups, perturbation)
        Vg = self._simplify(first_basis.conj().T @ perturbation @ first_basis)
        U = first_basis.copy()
        branches = []
        all_columns = set(range(self.n_basis))

        for block, (columns, energy, first_order, block_offset) in enumerate(block_data):
            outside = sorted(all_columns.difference(columns))
            B = Vg[np.ix_(outside, columns)]
            gaps = energy - np.asarray(good_energies, dtype=self.e0.dtype)[outside]
            X = B / gaps[:, None]
            W2 = self._simplify(B.conj().T @ X)
            v_scale = w2_scale = 0.0
            if not self.exact:
                # 公共一阶位移不参与定标。B 提供标量块附近的独立参考尺度。
                block_v = Vg[np.ix_(columns, columns)]
                mean = np.trace(block_v).real / len(columns)
                b_norm = np.linalg.norm(B)
                v_scale = max(
                    float(np.linalg.norm(block_v - mean * np.eye(len(columns)))),
                    float(b_norm),
                )
                w2_scale = float(b_norm * np.linalg.norm(X))
            first_groups = self._partition(first_order, scale_floor=v_scale)
            rotation = np.eye(len(columns), dtype=U.dtype)
            second_order = W2.diagonal().copy() if self.exact else W2.diagonal().real.copy()
            for group in first_groups:
                if len(group) > 1:
                    values, vectors = self._eigensystem(W2[np.ix_(group, group)])
                    rotation[np.ix_(group, group)] = vectors
                    second_order[list(group)] = values
            W2 = self._simplify(rotation.conj().T @ W2 @ rotation)
            X = self._simplify(X @ rotation)
            U[:, columns] = self._simplify(first_basis[:, columns] @ rotation)

            C1 = None
            if not energy_only:
                C = None
                if any(len(g) > 1 for g in first_groups):
                    C = Vg[np.ix_(outside, outside)] - block_offset * np.eye(len(outside), dtype=U.dtype)
                within = self._state_corrections(
                    block, first_order, first_groups, second_order, W2, X, C, w2_scale
                )
                C1 = self._simplify(first_basis[:, outside] @ X + U[:, columns] @ within)
            for local, alpha in enumerate(columns):
                branches.append(DegenerateBranch(
                    good_index=alpha,
                    block=block,
                    e0=energy,
                    dE1=first_order[local] + block_offset + offset,
                    dE2=second_order[local],
                    c0=U[:, alpha].copy(),
                    c1=None if C1 is None else C1[:, local].copy(),
                ))
        return DegeneratePerturbationResult(branches=branches, U=U, groups=groups)
