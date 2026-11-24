"""
qmpert.symmetry
===============

选择定则（selection rules）框架：把"群论/守恒量决定的禁戒矩阵元"变成

1. 一个布尔掩码 ``matrix_mask(N)`` —— 用于校验 V、并用稀疏结构加速；
2. 一个**先验**的零点阶预测 ``predicted_zero_orders(state, max_order)``。

为什么对称性能判断"某阶必然为零"
--------------------------------
能量修正 E_n⁽ᵐ⁾ = ⟨n|V|n⁽ᵐ⁻¹⁾⟩。若 |n⟩ 是某个守恒量（例如宇称）的本征态，
每作用一次 V 就把该量子数改变 Δ，则在 |n⁽ᵐ⁻¹⁾⟩ 中该量子数相对 |n⟩ 偏移了
(m-1)Δ。⟨n|V|n⁽ᵐ⁻¹⁾⟩ 非零要求再作用一次 V 后回到原量子数：

    m·Δ ≡ 0  (mod 规则)

* 宇称：p_V = -1 → 只有 **偶数阶** 能量修正非零（m·1 为偶数）；
  p_V = +1 → 无限制。
* 平移/宇称破缺的线性势 −Fx 是 p_V = -1 的特例，且微扰级数在二阶**精确终止**。

扩展方式
--------
继承 :class:`Symmetry`，实现 ``matrix_mask``（必须）与 ``predicted_zero_orders``
（可选，用于先验判断），即可接入任意点群/空间群/粒子数守恒等选择定则，
求解器本身无需改动。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np

try:                                        # 稀疏支持是可选的
    from scipy import sparse as _sp

    def _issparse(a) -> bool:
        return _sp.issparse(a)

except ImportError:                         # pragma: no cover
    def _issparse(a) -> bool:
        return False


__all__ = [
    "Symmetry",
    "NoSymmetry",
    "ParitySymmetry",
    "DeltaSymmetry",
    "QuantumNumberSymmetry",
    "CompositeSymmetry",
]


class Symmetry(ABC):
    """选择定则接口。

    子类必须实现 :meth:`matrix_mask`。若还能实现 :meth:`entry_allowed`
    （逐元判断，O(nnz)），请置 ``entrywise = True``——这样在稀疏矩阵、
    上万基态时求解器不必生成 N×N 的稠密掩码。
    """

    name: str = "abstract"
    entrywise: bool = False

    @abstractmethod
    def matrix_mask(self, n_basis: int) -> np.ndarray:
        """返回 (N, N) 布尔矩阵，True 表示该矩阵元被对称性允许。"""

    def entry_allowed(self, rows, cols):
        """逐元判断 ``(rows[i], cols[i])`` 是否被允许（向量化，可广播）。

        默认回退到 :meth:`matrix_mask`；子类若覆写，请同时置 ``entrywise = True``。
        """
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        n = int(np.max(np.concatenate([rows.ravel(), cols.ravel()]))) + 1
        return self.matrix_mask(n)[rows, cols]

    def predicted_zero_orders(self, state: int, max_order: int) -> set[int]:
        """先验判定：哪些阶 m 的能量修正必然为 0。默认不作预测。"""
        return set()

    # -- 工具 ---------------------------------------------------------- #
    def validate(self, v, rtol: float = 1e-10) -> float:
        """返回禁戒元的最大绝对值；≈0 说明 V 与该选择定则自洽。

        稀疏输入走逐元路径，不展开成稠密矩阵。
        """
        if _issparse(v) and self.entrywise:
            coo = v.tocoo()
            allowed = self.entry_allowed(coo.row, coo.col)
            bad = coo.data[~np.asarray(allowed, dtype=bool)]
            return float(np.max(np.abs(bad))) if bad.size else 0.0
        mask = self.matrix_mask(v.shape[0])
        forbidden = np.asarray(v)[~mask]
        return float(np.max(np.abs(forbidden))) if forbidden.size else 0.0

    def summary(self) -> str:
        return self.name

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.name}>"


class NoSymmetry(Symmetry):
    name = "none"

    def matrix_mask(self, n_basis: int) -> np.ndarray:
        return np.ones((n_basis, n_basis), dtype=bool)


class ParitySymmetry(Symmetry):
    """一维/中心对称体系的宇称选择定则。

    Parameters
    ----------
    state_parities : (N,) array of ±1
        每个基态的宇称本征值。
    perturbation_parity : ±1
        微扰算符 V 的宇称。
    """

    def __init__(self, state_parities: Sequence[int], perturbation_parity: int = -1) -> None:
        self.p = np.asarray(state_parities).astype(int)
        self.pv = int(perturbation_parity)
        if self.pv not in (-1, 1):
            raise ValueError("perturbation_parity 必须为 +1 或 -1")
        self.name = f"parity (p_V={self.pv:+d})"
        self.entrywise = True

    def matrix_mask(self, n_basis: int) -> np.ndarray:
        p = self.p[:n_basis]
        return (p[:, None] * p[None, :] * self.pv) > 0

    def entry_allowed(self, rows, cols):
        return (self.p[rows] * self.p[cols] * self.pv) > 0

    def predicted_zero_orders(self, state: int, max_order: int) -> set[int]:
        if self.pv == -1:                       # 奇宇称微扰 → 奇阶能量修正为零
            return {m for m in range(1, max_order + 1) if m % 2 == 1}
        return set()

    def summary(self) -> str:
        return (
            "宇称守恒：⟨k|V|n⟩≠0 需 p_k·p_n·p_V=+1；"
            + ("奇宇称微扰 ⇒ 奇阶能量修正恒为零" if self.pv == -1 else "偶宇称微扰 ⇒ 无阶数限制")
        )


class DeltaSymmetry(Symmetry):
    """"指标差"型选择定则：⟨k|V|n⟩≠0 仅当 (k-n) ∈ allowed_deltas。

    适用于谐振子型基（x^q 只耦合 Δn = ±q, ±(q-2), …）。
    """

    def __init__(self, allowed_deltas: Iterable[int], name: str = "") -> None:
        self.deltas = frozenset(int(d) for d in allowed_deltas)
        self.name = name or f"Δn ∈ {sorted(self.deltas)}"
        self.entrywise = True

    def matrix_mask(self, n_basis: int) -> np.ndarray:
        idx = np.arange(n_basis)
        diff = idx[:, None] - idx[None, :]
        return np.isin(diff, np.fromiter(self.deltas, dtype=diff.dtype, count=len(self.deltas)))

    def entry_allowed(self, rows, cols):
        return np.isin(np.asarray(rows) - np.asarray(cols), list(self.deltas))

    def predicted_zero_orders(self, state: int, max_order: int) -> set[int]:
        if 0 not in self.deltas and all(d % 2 != 0 for d in self.deltas):
            # 所有允许的 Δ 都是奇数 ⇒ 偏移 m 次后奇偶性为 (-1)^m
            return {m for m in range(1, max_order + 1) if m % 2 == 1}
        return set()


class QuantumNumberSymmetry(Symmetry):
    """一般量子数守恒：labels[k] - labels[n] ∈ allowed_deltas。

    例：2D 各向异性谐振子中 n_x、n_y 分别守恒；角动量 m 守恒等。
    """

    def __init__(self, labels: Sequence, allowed_deltas: Iterable, name: str = "") -> None:
        self.labels = np.asarray(labels)
        self.deltas = frozenset(allowed_deltas)
        self.name = name or f"Δq ∈ {sorted(self.deltas)}"
        self.entrywise = True

    def matrix_mask(self, n_basis: int) -> np.ndarray:
        q = self.labels[:n_basis]
        diff = q[:, None] - q[None, :]
        return np.isin(diff, list(self.deltas))

    def entry_allowed(self, rows, cols):
        return np.isin(self.labels[rows] - self.labels[cols], list(self.deltas))


class CompositeSymmetry(Symmetry):
    """多个守恒量同时成立（掩码取交，零点预测取并）。"""

    def __init__(self, parts: Sequence[Symmetry], name: str = "") -> None:
        self.parts = list(parts)
        self.name = name or " ∧ ".join(p.name for p in self.parts)
        self.entrywise = all(getattr(p, "entrywise", False) for p in self.parts)

    def matrix_mask(self, n_basis: int) -> np.ndarray:
        mask = np.ones((n_basis, n_basis), dtype=bool)
        for p in self.parts:
            mask &= p.matrix_mask(n_basis)
        return mask

    def entry_allowed(self, rows, cols):
        ok = np.ones(np.broadcast(rows, cols).shape, dtype=bool)
        for p in self.parts:
            ok &= np.asarray(p.entry_allowed(rows, cols), dtype=bool)
        return ok

    def predicted_zero_orders(self, state: int, max_order: int) -> set[int]:
        z: set[int] = set()
        for p in self.parts:
            z |= p.predicted_zero_orders(state, max_order)
        return z
