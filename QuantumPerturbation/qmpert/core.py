"""
qmpert.core
===========

任意阶非简并 Rayleigh-Schrödinger 微扰论（RSPT）求解器。

数学约定
--------
    H(λ) = H₀ + λ V ,      H₀|n⟩ = E_n⁽⁰⁾|n⟩ ,  谱非简并

    中间归一化 (intermediate normalization)  ⟨n⁽⁰⁾|n(λ)⟩ = 1 ，故
        E_n(λ) = Σ_{m≥0} λ^m E_n⁽ᵐ⁾ ,      E_n⁽⁰⁾ ≡ E_n⁽⁰⁾
        |n(λ)⟩ = Σ_{m≥0} λ^m |n⁽ᵐ⁾⟩ ,      ⟨n⁽⁰⁾|n⁽ᵐ⁾⟩ = δ_{m0}

把展开代入 (H₀ + λV)|n⟩ = E_n|n⟩ 并按 λ 逐阶比较，得到两条**自洽递推**
（对 m ≥ 1，k ≠ n）：

    E_n⁽ᵐ⁾   = ⟨n⁽⁰⁾| V |n⁽ᵐ⁻¹⟩                                  (能量)
    ⟨k⁽⁰⁾|n⁽ᵐ⁾⟩ = [ -⟨k⁽⁰⁾|V|n⁽ᵐ⁻¹⟩ + Σ_{p=1}^{m-1} E_n⁽ᵖ⁾⟨k⁽⁰⁾|n⁽ᵐ⁻ᵖ⁾⟩ ]
                  / (E_k⁽⁰⁾ - E_n⁽⁰⁾)                             (波函数)
    ⟨n⁽⁰⁾|n⁽ᵐ⁾⟩ = 0

递推只用到上一阶的态矢与全部低阶能量，因此对阶数 m **没有上限**；
每阶一次矩阵-向量乘，再加 O(mN) 的反馈求和；到 M 阶共 O(MN²+M²N)，
存储矩阵之外另需 O(MN)。

扩展点
------
* ``symmetry`` 参数接受 :class:`qmpert.symmetry.Symmetry` 实例，用群论/量数
  选择定则把 V 的"禁戒"矩阵元结构清零（并可在求解前校验），同时给出
  "哪些阶必然为零"的先验预测。
* ``backend`` 支持 float / complex / 精确有理数（Fraction 对象数组），
  便于得到 E_n⁽ᵐ⁾ 的解析分数值。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Optional, Sequence

import numpy as np

try:                                        # 稀疏矩阵支持（可选依赖）
    from scipy import sparse as _sp

    def _issparse(a) -> bool:
        return _sp.issparse(a)

except ImportError:                         # pragma: no cover
    _sp = None

    def _issparse(a) -> bool:
        return False


__all__ = [
    "PerturbationTheory",
    "PerturbationSeries",
    "DegenerateSpectrumError",
]


class DegenerateSpectrumError(ValueError):
    """目标态与其它态（近）简并时抛出：此时非简并 RSPT 的分母发散。"""


# --------------------------------------------------------------------------- #
#  结果容器
# --------------------------------------------------------------------------- #
@dataclass
class PerturbationSeries:
    """一个本征态（一个 ``state``）的微扰级数结果。"""

    state: int
    e0: np.ndarray                    # 未微扰谱 (N,)
    dE: np.ndarray                    # 能量修正 (M+1,)，dE[0] = 0
    coeffs: np.ndarray                # 态矢系数 (M+1, N)，coeffs[0] = e_n
    symmetry_name: str = "none"
    zeros_predicted: frozenset = frozenset()
    _energy_bounds: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    # ---------------- 基本属性 ----------------
    @property
    def max_order(self) -> int:
        return len(self.dE) - 1

    @property
    def unperturbed_energy(self) -> float:
        return float(np.real(self.e0[self.state]))

    def _order(self, order: Optional[int]) -> int:
        m = self.max_order if order is None else int(order)
        if not 0 <= m <= self.max_order:
            raise ValueError(f"order 必须在 0 到 {self.max_order} 之间")
        return m

    # ---------------- 求值 ----------------
    def energy(self, lam: float, order: Optional[int] = None):
        """截断到 ``order`` 阶的微扰能量 Σ_{m≤order} λ^m E⁽ᵐ⁾。"""
        result = 0
        for k in range(self._order(order), 0, -1):
            result = lam * (result + self.dE[k])
        return self.e0[self.state] + result

    def energy_exact(self, lam=0, order: Optional[int] = None) -> object:
        """精确截断能量。纯有理数返回 Fraction，含符号项时保留 SymPy 表达式。"""
        if self.dE.dtype != object:
            raise TypeError("该级数是浮点结果；请用 Fraction/sympy 类型的 e0、V 构造")
        m_max = self._order(order)
        lam = _as_exact(lam)
        result = Fraction(0)
        for m in range(m_max, 0, -1):
            result = lam * (result + _as_exact(self.dE[m]))
        return _as_exact(self.e0[self.state]) + result

    def state_vector(self, lam: float, order: Optional[int] = None) -> np.ndarray:
        """截断到 ``order`` 阶的态矢（未微扰基下系数）。"""
        m = self._order(order)
        vec = self.coeffs[m].astype(np.result_type(self.coeffs.dtype, type(lam)))
        for k in range(m - 1, -1, -1):
            vec = self.coeffs[k] + lam * vec
        return vec

    def normalized_coeffs(self) -> np.ndarray:
        """返回逐阶范数归一化 ``u_m``，使 ``Σ_k ⟨u_k|u_{m-k}⟩ = δ_m0``。

        相位取通常约定：归一化因子为实数且零阶系数不变。计算只使用
        已有系数和精确内积，因此 object 后端仍保持 Fraction/sympy 算术。
        """
        u = np.empty_like(self.coeffs)
        u[0] = self.coeffs[0]
        for m in range(1, self.max_order + 1):
            a = self.coeffs[m, self.state]  # m=1 时即为精确的零
            if m > 1:
                if self.coeffs.dtype == object:
                    overlap = sum(
                        (
                            u[k, i].conjugate() * u[m - k, i]
                            for k in range(1, m)
                            for i in range(u.shape[1])
                        ),
                        0,
                    )
                else:
                    overlap = sum(
                        (np.vdot(u[k], u[m - k]) for k in range(1, m)), 0
                    )
                if hasattr(overlap, "as_real_imag"):
                    real_overlap = overlap.as_real_imag()[0]
                else:
                    real_overlap = overlap.real
                a = -real_overlap / 2

            u[m] = self.coeffs[m]
            for r in range(1, m):
                u[m] += u[r, self.state] * self.coeffs[m - r]
            u[m, self.state] += a
        return u

    def normalized_state(self, lam: float, order: Optional[int] = None) -> np.ndarray:
        """把截断态矢归一化（微扰展开本身不是归一化的）。"""
        v = self.state_vector(lam, order)
        if v.dtype == object:
            import sympy as sp

            norm = sp.sqrt(sum(sp.conjugate(x) * x for x in v))
            return np.array([x / norm for x in v], dtype=object)
        nrm = np.linalg.norm(v)
        return v / nrm if nrm > 0 else v

    # ---------------- 诊断 ----------------
    def measured_zero_orders(self, scale: Optional[float] = None) -> set[int]:
        """精确后端判零，数值后端默认按每阶能量内积的绝对值上界定标。"""
        if self.dE.dtype == object and scale is None:
            return {m for m in range(1, self.max_order + 1) if self.dE[m] == 0}
        mag = np.abs(self.dE[1:])
        if scale is None and self._energy_bounds is not None:
            bounds = self._energy_bounds
            eps = np.finfo(bounds.dtype).eps
            zero = (mag == 0) | (np.isfinite(bounds) & (mag <= 64 * eps * bounds))
            return set((np.flatnonzero(zero) + 1).tolist())
        ref = scale if scale is not None else mag.max(initial=0.0)
        thr = 1e-13 * max(ref, 1e-300)
        return {m for m, value in enumerate(mag, start=1) if value <= thr}

    def nonzero_orders(self, scale: Optional[float] = None) -> list[int]:
        z = self.measured_zero_orders(scale)
        return [m for m in range(1, self.max_order + 1) if m not in z]

    def terms(self) -> list[tuple[int, object]]:
        return [(m, self.dE[m]) for m in range(self.max_order + 1)]

    def __repr__(self) -> str:  # pragma: no cover
        z = sorted(self.measured_zero_orders())
        return (
            f"<PerturbationSeries n={self.state} order={self.max_order} "
            f"E0={self.unperturbed_energy:.6g} zero_orders={z}>"
        )


# --------------------------------------------------------------------------- #
#  矩阵–向量乘：精确后端（object dtype）走纯 Python
# --------------------------------------------------------------------------- #
def _as_exact(x) -> object:
    """有理数统一为 Fraction，保留其他 SymPy 表达式。"""
    if isinstance(x, Fraction):
        return x
    for a, b in (("p", "q"), ("numerator", "denominator")):
        num, den = getattr(x, a, None), getattr(x, b, None)
        if num is not None and den is not None:
            return Fraction(int(num), int(den))
    try:
        return Fraction(x)
    except TypeError:
        import sympy as sp

        if isinstance(x, sp.Expr):
            return x
        raise


def _matvec(A, x):
    """``A @ x``，但对 ``object`` dtype（Fraction / sympy）保持精确算术。

    object 路径跳过零元，并兼容 SciPy 不支持 object 向量的稀疏乘法。
    """
    if x.dtype != object:
        return A @ x
    n = x.size
    out = np.empty(n, dtype=object)
    if _issparse(A):
        A = A.tocsr()
        for i in range(n):
            lo, hi = A.indptr[i], A.indptr[i + 1]
            total = 0
            for idx in range(lo, hi):
                total += A.data[idx] * x[A.indices[idx]]
            out[i] = total
        return out
    for i in range(n):
        row = A[i]
        total = 0
        for j in range(n):
            a = row[j]
            if a:
                total += a * x[j]
        out[i] = total
    return out


# --------------------------------------------------------------------------- #
#  选择定则：稠密 / 稀疏两条路径
# --------------------------------------------------------------------------- #
def _apply_selection_rule(v, symmetry, n_basis: int, enforce: bool):
    """按选择定则识别（并可清零）禁戒元，返回 ``(v_out, violation)``。

    稀疏输入走**结构路径**：只遍历已存储的 nnz 个元，不生成 N×N 掩码；
    这正是上万维基组可行的原因。
    """
    if _issparse(v):
        coo = v.tocoo()
        if getattr(symmetry, "entrywise", False):
            keep = np.asarray(symmetry.entry_allowed(coo.row, coo.col), dtype=bool)
        else:
            keep = np.asarray(symmetry.matrix_mask(n_basis), dtype=bool)[coo.row, coo.col]
        bad = coo.data[~keep]
        violation = float(np.max(np.abs(bad))) if bad.size else 0.0
        if not enforce or bool(np.all(keep)):
            return v, violation
        out = _sp.coo_matrix(
            (coo.data[keep], (coo.row[keep], coo.col[keep])), shape=(n_basis, n_basis)
        ).tocsr()
        out.sum_duplicates()
        out.eliminate_zeros()
        return out, violation

    mask = np.asarray(symmetry.matrix_mask(n_basis), dtype=bool)
    forbidden = v[~mask]
    violation = float(np.max(np.abs(forbidden))) if forbidden.size else 0.0
    if not enforce:
        return v, violation
    return np.where(mask, v, np.zeros((), dtype=v.dtype)), violation


# --------------------------------------------------------------------------- #
#  求解器
# --------------------------------------------------------------------------- #
def _warn_if_nonhermitian(V) -> None:
    """比较 V−V† 的最大元与 V 的 Frobenius 范数，稀疏矩阵仅遍历存储项。"""
    sparse = _issparse(V)
    scaled = V.astype(np.result_type(V.dtype, np.float64))
    entries = scaled.data if sparse else scaled
    is_complex = np.iscomplexobj(entries)
    scale = np.max(np.abs(entries.real), initial=0.0)
    if is_complex:
        scale = max(scale, np.max(np.abs(entries.imag), initial=0.0))
    if scale == 0 or not np.isfinite(scale):
        return
    # 先缩放再作差和求范数，避免在极大或极小能标下溢出或下溢。
    entries.real[...] /= scale
    if is_complex:
        entries.imag[...] /= scale
    if sparse:
        scaled.sum_duplicates()
        entries = scaled.data
    diff = scaled - scaled.conj().T
    asymmetry = np.max(np.abs(diff.data if sparse else diff), initial=0.0)
    if asymmetry == 0:
        return
    relative = asymmetry / np.linalg.norm(entries.ravel())
    if relative > 1e-10:
        warnings.warn(
            f"V 的相对厄米性偏差为 {relative:.2e}（最大元偏差除以 Frobenius 范数），"
            "微扰结果可能不可靠。",
            RuntimeWarning,
            stacklevel=3,
        )


class PerturbationTheory:
    """非简并微扰论引擎。

    Parameters
    ----------
    h0_energies : (N,) array
        未微扰哈密顿量 H₀ 的本征值。
    v_matrix : (N, N) array
        V 在 H₀ 本征基下的厄米矩阵。数值后端在应用选择定则后检查厄米性。
    symmetry : Symmetry or None
        选择定则；给出后会（a）校验 V 的禁戒元是否为零，（b）可选地强制清零。
    enforce_symmetry : bool
        为 True 时按选择定则把禁戒元清零（物理上等价，但可利用稀疏性加速）。
    degeneracy_tol : float
        近简并判据（相对量级）。

    说明
    ----
    非简并性只需对目标态成立；数值后端用谱宽定标近简并容差，
    object 后端检查精确相等。
    """

    def __init__(
        self,
        h0_energies: Sequence,
        v_matrix: Sequence,
        symmetry=None,
        enforce_symmetry: bool = True,
        degeneracy_tol: float = 1e-10,
        name: str = "",
    ) -> None:
        self.e0 = np.asarray(h0_energies)
        if self.e0.ndim != 1 or self.e0.size == 0:
            raise ValueError("h0_energies 必须是非空 1D 数组")
        n_basis = self.e0.size
        self.sparse = _issparse(v_matrix)
        self.v_raw = v_matrix.tocsr() if self.sparse else np.asarray(v_matrix)
        if self.e0.dtype != object:
            if np.any(np.imag(self.e0) != 0) or not np.all(np.isfinite(self.e0)):
                raise ValueError("h0_energies 必须是有限实数")
            real = np.real(self.e0)
            if self.v_raw.dtype == object and real.dtype.kind == "f":
                # 保留输入浮点数的精确二进制值，避免能隙除法降回浮点。
                self.e0 = np.array([Fraction(*x.as_integer_ratio()) for x in real], dtype=object)
            else:
                exact_int = self.v_raw.dtype == object and real.dtype.kind in "iub"
                self.e0 = real.astype(object if exact_int else np.result_type(real.dtype, np.float64))
        if self.v_raw.shape != (n_basis, n_basis):
            raise ValueError(
                f"v_matrix 形状应为 {(n_basis, n_basis)}，收到 {self.v_raw.shape}"
            )
        self.symmetry = symmetry
        self.name = name
        if not np.isfinite(degeneracy_tol) or degeneracy_tol < 0:
            raise ValueError("degeneracy_tol 必须是有限非负数")
        self.degeneracy_tol = degeneracy_tol
        self.enforce_symmetry = enforce_symmetry
        self.mask = None
        self.v = self.v_raw
        self.violation = 0.0
        self.matvecs = 0
        if symmetry is not None:
            self.v, self.violation = _apply_selection_rule(
                self.v_raw, symmetry, n_basis, enforce_symmetry
            )
            if not self.sparse:
                self.mask = np.asarray(symmetry.matrix_mask(n_basis), dtype=bool)
        if self.e0.dtype != object and self.v.dtype != object:
            _warn_if_nonhermitian(self.v)

    # ------------------------------------------------------------------ #
    def _check_nondegenerate(self, n: int, gaps: np.ndarray) -> None:
        if gaps.dtype == object:
            candidates = (
                k for k, gap in enumerate(gaps)
                if k != n and (gap == 0 or getattr(gap, "is_zero", None) is True)
            )
            k = next(candidates, None)
        else:
            cutoff = self.degeneracy_tol * np.ptp(self.e0)
            gap_magnitudes = np.abs(gaps)
            gap_magnitudes[n] = np.inf
            nearest = int(np.argmin(gap_magnitudes))
            k = nearest if gap_magnitudes[nearest] <= cutoff else None
        if k is not None:
            raise DegenerateSpectrumError(
                f"态 {n} 与态 {k} 的能隙为 {gaps[k]}，触及简并判据；请使用简并微扰论。"
            )

    # ------------------------------------------------------------------ #
    def solve(self, state: int, max_order: int) -> PerturbationSeries:
        """计算态 ``state`` 直到 ``max_order`` 阶的能量与态矢修正。"""
        n = int(state)
        N = self.e0.size
        M = int(max_order)
        if not -N <= n < N:
            raise IndexError(f"state 超出基组范围（N={N}）")
        n %= N
        if M < 0:
            raise ValueError("max_order 必须 ≥ 0")

        dtype = np.result_type(self.e0.dtype, self.v.dtype, np.float64)
        gaps = (self.e0 - self.e0[n]).astype(dtype, copy=True)
        # 保留原有能谱数据类型对简并判据的语义；object 的整数分母也必须精确。
        self._check_nondegenerate(n, self.e0 - self.e0[n])
        zero = Fraction(0) if dtype == object else 0.0
        c = np.full((M + 1, N), zero, dtype=dtype)
        c[0, n] = Fraction(1) if dtype == object else 1
        dE = np.full(M + 1, zero, dtype=dtype)
        if dtype == object:
            gaps = np.array([
                Fraction(int(gap)) if isinstance(gap, (int, np.integer)) else gap
                for gap in gaps
            ], dtype=object)
        gaps[n] = 1  # 目标态分量在下方按中间归一化单独设零
        V = self.v
        energy_bounds = None
        if dtype != object:
            if self.sparse:
                start, end = V.indptr[n : n + 2]
                columns = V.indices[start:end]
                row = V.data[start:end]
            else:
                columns = slice(None)
                row = V[n]
            weights = np.abs(row.astype(dtype, copy=False))
            energy_bounds = np.empty(M, dtype=dE.real.dtype)

        self.matvecs = 0
        for m in range(1, M + 1):
            vc = _matvec(V, c[m - 1])         # 一次矩阵-向量乘
            self.matvecs += 1
            dE[m] = vc[n]                     # E⁽ᵐ⁾ = ⟨n|V|n⁽ᵐ⁻¹⟩
            if energy_bounds is not None:
                energy_bounds[m - 1] = weights @ np.abs(c[m - 1, columns])
            rhs = -vc
            for p in range(1, m):             # 低阶能量的反馈项
                rhs = rhs + dE[p] * c[m - p]
            c[m] = rhs / gaps
            c[m, n] = zero

        zeros = frozenset()
        if self.symmetry is not None:
            zeros = frozenset(self.symmetry.predicted_zero_orders(n, M))
        return PerturbationSeries(
            state=n,
            e0=self.e0,
            dE=dE,
            coeffs=c,
            symmetry_name=getattr(self.symmetry, "name", "none"),
            zeros_predicted=zeros,
            _energy_bounds=energy_bounds,
        )

    # ------------------------------------------------------------------ #
    def solve_many(self, states: Iterable[int], max_order: int) -> list[PerturbationSeries]:
        return [self.solve(n, max_order) for n in states]

    def diagnostics(self) -> dict:
        N = int(self.e0.size)
        nnz_raw = int(self.v_raw.nnz if self.sparse else np.count_nonzero(self.v_raw))
        nnz_used = int(self.v.nnz if self.sparse else np.count_nonzero(self.v))
        d = {
            "name": self.name,
            "n_basis": N,
            "sparse": bool(self.sparse),
            "symmetry": getattr(self.symmetry, "name", "none"),
            "symmetry_violation": self.violation,
            "nnz_raw": nnz_raw,
            "nnz_used": nnz_used,
            "matvecs": int(self.matvecs),
        }
        d["nnz_ratio"] = nnz_used / max(nnz_raw, 1)
        d["storage_ratio"] = nnz_used / max(N * N, 1)
        d["dense_mb"] = N * N * 8 / 1e6
        # CSR: 数据 + 列索引（4B）+ 行指针
        d["stored_mb"] = (nnz_used * 12 + (N + 1) * 4) / 1e6 if self.sparse else N * N * 8 / 1e6
        d["density"] = d["storage_ratio"]
        return d

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PerturbationTheory {self.name!r} N={self.e0.size} sym={getattr(self.symmetry,'name','none')}>"
