"""
qmpert.resum
============

发散级数的重求和（resummation）。

RSPT 的能量级数对绝大多数模型都是**零收敛半径的渐近级数**：系数按
``m!`` 增长，截断到最佳阶 ``m* ≈ 1/(cλ)`` 后误差不再下降。本模块提供两种
把"发散的泰勒级数"变成"可用的数值结果"的标准手段：

1. **Padé 逼近**  ``[L/M]``
   ``f(x) ≈ P_L(x)/Q_M(x)``，其中 ``P, Q`` 由"前 ``L+M`` 阶 Taylor 系数
   完全匹配"唯一确定。Padé 逼近能跨越 Taylor 级数的收敛半径，把极点/分支点
   之外的信息也编码进来——这是它比截断部分和强得多的原因。

2. **Borel–Padé**
   先作 Borel 变换 ``B(t) = Σ a_m t^m / m!``（阶乘增长被压成几何增长），
   再对 ``B`` 作 Padé 逼近，最后用 Laplace 积分还原：
   ``f(λ) = ∫_0^∞ e^{-t} B(λt) dt``。
   它对阶乘发散特别有效；代价是 Padé 的伪极点若落在正实轴上会破坏积分，
   因此实现里带有极点筛查。

数值说明
--------
Padé 的线性方程组条件数随阶数急剧恶化（系数本身跨上百个数量级），双精度
求解在 ``L ≳ 16`` 时已不可靠。因此构造默认走 ``mpmath`` 高精度（位数随阶数
自动放大），再把系数舍入回 ``float``；求值仍在双精度下进行——实测两者在
``λ ≲ 1`` 内逐位一致。若无 ``mpmath`` 则退回双精度并给出警告。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import factorial
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.integrate import quad

try:  # mpmath 随 sympy 一同安装，属于可选依赖
    import mpmath as _mp
except Exception:  # pragma: no cover
    _mp = None

__all__ = [
    "PadeApproximant",
    "PadeTable",
    "BorelPade",
    "pade",
    "borel_pade",
]


# --------------------------------------------------------------------------- #
#  核心：由 Taylor 系数构造 [L/M]
# --------------------------------------------------------------------------- #
@dataclass
class PadeApproximant:
    """``[L/M]`` Padé 逼近 ``P(x)/Q(x)``（系数按升幂存放，``Q[0] = 1``）。"""

    p: np.ndarray
    q: np.ndarray
    L: int
    M: int
    residual_mp: float = float("nan")   # 高精度下的匹配残差（构造时算好）
    _p_mp: list | None = None           # 高精度系数（若有，求值也走高精度）
    _q_mp: list | None = None

    def __call__(self, x):
        if self._p_mp is not None:
            scalar = np.ndim(x) == 0
            arr = np.atleast_1d(np.asarray(x, dtype=float))
            old = _mp.mp.dps
            try:
                _mp.mp.dps = 50
                out = [
                    float(
                        _mp.polyval(self._p_mp[::-1], _mp.mpf(float(t)))
                        / _mp.polyval(self._q_mp[::-1], _mp.mpf(float(t)))
                    )
                    for t in arr
                ]
            finally:
                _mp.mp.dps = old
            return out[0] if scalar else np.array(out).reshape(np.shape(x))
        x = np.asarray(x, dtype=float)
        num = np.polyval(self.p[::-1], x)
        den = np.polyval(self.q[::-1], x)
        return num / den

    def poles(self) -> np.ndarray:
        """分母的根（Padé 极点），按 |z| 升序。"""
        if self.M == 0:
            return np.array([])
        r = np.roots(self.q[::-1])
        return r[np.argsort(np.abs(r))]

    def residuals(self, coeffs: Sequence[float]) -> float:
        """``Q·Σc − P`` 在匹配阶内的**相对**残差（校验代数恒等式）。

        若构造时走高精度，直接返回当时算好的值：把系数舍入到 double 后再重算，
        高阶情形会因分母系数的巨大动态范围而失真（实测可达 O(1)）。
        """
        if self.residual_mp == self.residual_mp:      # 非 NaN
            return self.residual_mp
        c = np.asarray(coeffs, dtype=float)
        n = self.L + self.M
        prod = np.convolve(self.q, c[: n + 1])[: n + 1]
        diff = prod.copy()
        diff[: self.p.size] -= self.p
        scale = np.max(np.abs(prod))
        return float(np.max(np.abs(diff)) / scale) if scale else float(np.max(np.abs(diff)))

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Padé [{self.L}/{self.M}]>"


def _pade_float(c: np.ndarray, L: int, M: int) -> tuple[np.ndarray, np.ndarray, float]:
    if M == 0:
        q = np.array([1.0])
    else:
        A = np.empty((M, M))
        rhs = np.empty(M)
        for j in range(1, M + 1):
            for i in range(1, M + 1):
                A[j - 1, i - 1] = c[L + j - i]
            rhs[j - 1] = -c[L + j]
        scale = np.max(np.abs(A), axis=1)
        scale[scale == 0.0] = 1.0
        q = np.concatenate(([1.0], np.linalg.solve(A / scale[:, None], rhs / scale)))
    p = np.array(
        [sum(c[j - i] * q[i] for i in range(min(j, M) + 1)) for j in range(L + 1)]
    )
    n = L + M
    prod = np.convolve(q, c[: n + 1])[: n + 1]
    diff = prod.copy()
    diff[: p.size] -= p
    scale = np.max(np.abs(prod)) if prod.size else 1.0
    resid = float(np.max(np.abs(diff)) / scale) if scale else float(np.max(np.abs(diff)))
    return p, q, resid


def _pade_mp(c: np.ndarray, L: int, M: int, dps: int) -> tuple[np.ndarray, np.ndarray, float, list, list]:
    old = _mp.mp.dps
    try:
        _mp.mp.dps = dps
        a = [_mp.mpf(float(x)) for x in c[: L + M + 1]]
        p, q = _mp.pade(a, L, M)
        # 在**高精度下**检验 Q·Σc − P 是否为 0；舍入到 double 后该恒等式
        # 会因 q 的动态范围而失真，故残差必须在此处计算
        prod = [sum(q[i] * a[j - i] for i in range(min(j, M) + 1)) for j in range(L + M + 1)]
        diff = [prod[j] - (p[j] if j <= L else 0) for j in range(L + M + 1)]
        scale = max(abs(x) for x in prod) or 1
        resid = float(max(abs(x) for x in diff) / scale)
        return (
            np.array([float(x) for x in p]),
            np.array([float(x) for x in q]),
            resid,
            list(p),
            list(q),
        )
    finally:
        _mp.mp.dps = old


def pade(coeffs: Sequence[float], L: int, M: int, high_precision: bool = True) -> PadeApproximant:
    """由 Taylor 系数 ``c[0..L+M]`` 构造 ``[L/M]`` Padé 逼近。

    高精度构造是默认行为：系数按阶乘增长，双精度线性求解在 ``L ≳ 16`` 时
    条件数超过 10¹⁶，结果不可信。
    """
    c = np.asarray(coeffs, dtype=float)
    L, M = int(L), int(M)
    if L < 0 or M < 0:
        raise ValueError("L, M 必须非负")
    if c.size < L + M + 1:
        raise ValueError(f"需要至少 {L+M+1} 个系数，收到 {c.size}")

    if high_precision and _mp is not None:
        # 位数按系数动态范围与阶数放大；上限避免极端情形过慢
        expo = int(np.ceil(np.log10(max(np.max(np.abs(c)), 1e-300))))
        dps = min(420, max(60, 10 * (L + M) // 4 + max(expo, 0) // 2 + 30))
        for attempt in (0, 1):          # 极端情况下再加一倍位数重试一次
            try:
                p, q, resid, p_mp, q_mp = _pade_mp(
                    c, L, M, dps if attempt == 0 else min(2 * dps, 600)
                )
                return PadeApproximant(p=p, q=q, L=L, M=M, residual_mp=resid,
                                       _p_mp=p_mp, _q_mp=q_mp)
            except ZeroDivisionError:
                continue
            except Exception:           # pragma: no cover
                break

    if high_precision and _mp is None:  # pragma: no cover
        warnings.warn("未安装 mpmath，Padé 构造退回双精度，高阶结果可能失真", RuntimeWarning)
    p, q, resid = _pade_float(c, L, M)
    return PadeApproximant(p=p, q=q, L=L, M=M, residual_mp=resid)


# --------------------------------------------------------------------------- #
#  对角 Padé 表：一次构造，多处求值
# --------------------------------------------------------------------------- #
@dataclass
class PadeTable:
    """对角 Padé 序列 ``[k/k]``，``k = 1 … K``。

    构造与 λ 无关，对一批 λ 求值时可复用同一张表。构造时跳过方程组奇异的阶。
    """

    coeffs: np.ndarray
    approx: dict = field(default_factory=dict)

    def __init__(self, coeffs: Sequence[float], max_order: Optional[int] = None,
                 high_precision: bool = True):
        c = np.asarray(coeffs, dtype=float)
        self.coeffs = c
        kmax = (c.size - 1) // 2 if max_order is None else min(int(max_order), (c.size - 1) // 2)
        self.max_order = kmax
        self.approx = {}
        for k in range(1, kmax + 1):
            try:
                self.approx[k] = pade(c, k, k, high_precision=high_precision)
            except np.linalg.LinAlgError:
                continue

    def at(self, lam: float, k: int) -> float:
        return float(self.approx[int(k)](lam))

    def best(self, lam: float, tol: float = 1e-12, k_min: int = 2) -> tuple[float, int, float]:
        """**不使用精确解**、仅凭序列自洽性选取阶数。

        按升序比较相邻的可用阶，返回首个达到相对误差阈值 ``tol`` 且
        ``k >= k_min`` 的 ``[k/k]``。若始终未达阈值，返回最后一组比较的结果。
        """
        if len(self.approx) < 2:
            raise ValueError("Padé 自洽选阶至少需要两个可用的对角逼近")
        prev = None
        last = None
        for k in sorted(self.approx):
            val = self.at(lam, k)
            if k >= k_min and prev is not None:
                delta = abs(val - prev) / max(abs(val), 1e-300)
                last = (val, k, delta)
                if delta < tol:
                    return val, k, delta
            prev = val
        if last is None:
            raise ValueError(f"没有可用于比较且满足 k ≥ {k_min} 的对角 Padé 逼近")
        return last

    def errors(self, lam: float, truth: float, k_min: int = 1) -> tuple[np.ndarray, np.ndarray]:
        ks = np.array([k for k in sorted(self.approx) if k >= k_min])
        vals = np.array([self.at(lam, k) for k in ks])
        return ks, np.abs(vals - truth)


# --------------------------------------------------------------------------- #
#  Borel–Padé
# --------------------------------------------------------------------------- #
def _borel_coeffs(coeffs: Sequence[float]) -> np.ndarray:
    """计算 Borel 系数。171 阶起用整数阶乘作除法，并将次正规结果置零。"""
    c = np.asarray(coeffs, dtype=float)
    result = np.where(np.isfinite(c), 0.0, c)
    for k in range(min(c.size, 171)):
        result[k] = c[k] / float(factorial(k))
    min_normal = np.finfo(float).tiny
    # 301! 起，即使用 float64 最大有限值作分子，商也小于最小正规数。
    for k in range(171, min(c.size, 301)):
        if c[k] == 0 or not np.isfinite(c[k]):
            continue
        numerator, denominator = float(c[k]).as_integer_ratio()
        value = numerator / (denominator * factorial(k))
        if abs(value) >= min_normal:
            result[k] = value
    return result


@dataclass
class BorelPade:
    """Borel 变换后的 ``[L/M]`` Padé，可用 Laplace 积分在任意 λ 求值。"""

    coeffs: np.ndarray
    approx: PadeApproximant
    L: int
    M: int

    def __init__(self, coeffs: Sequence[float], L: int, M: Optional[int] = None,
                 high_precision: bool = True):
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.L = int(L)
        self.M = int(L if M is None else M)
        self.approx = pade(_borel_coeffs(self.coeffs), self.L, self.M,
                           high_precision=high_precision)

    # ---------------- 极点筛查 ----------------
    def dangerous_poles(self, lam: float, arg_tol: float = 0.08,
                        t_max: float = 80.0) -> list[complex]:
        """积分路径 ``t ∈ (0, ∞)`` 附近（``|arg t| < arg_tol``）的极点。

        正实轴上的极点会让 Laplace 积分发散；即使极点只是"贴近"实轴，也会
        造成严重的数值误差。``t_max`` 之外因子 ``e^{−t}`` 已把贡献压到可忽略。
        """
        if self.M == 0 or lam == 0:
            return []
        t = self.approx.poles() / lam
        return [
            complex(z)
            for z in t
            if abs(z) < t_max and abs(np.angle(z)) < arg_tol
        ]

    def is_safe(self, lam: float, arg_tol: float = 0.08) -> bool:
        return not self.dangerous_poles(lam, arg_tol=arg_tol)

    # ---------------- 求值 ----------------
    def __call__(self, lam: float, epsabs: float = 1e-13, epsrel: float = 1e-11) -> float:
        if not self.is_safe(lam):
            bad = self.dangerous_poles(lam)
            raise ValueError(
                f"Borel–Padé [{self.L}/{self.M}] 在 λ={lam:g} 的积分路径附近有 "
                f"{len(bad)} 个极点（最近 |t|={min(abs(z) for z in bad):.3g}），结果不可信"
            )
        f = lambda t: np.exp(-t) * self.approx(lam * t)
        val, _ = quad(f, 0.0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=400)
        return float(val)


def borel_pade(coeffs: Sequence[float], lam: float, L: int, M: Optional[int] = None,
               screen: bool = True) -> float:
    """一次性调用的便捷接口（批量求值请用 :class:`BorelPade`）。"""
    b = BorelPade(coeffs, L, M)
    if not screen:
        f = lambda t: np.exp(-t) * b.approx(lam * t)
        return float(quad(f, 0.0, np.inf, limit=400)[0])
    return b(lam)
