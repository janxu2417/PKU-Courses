"""
qmpert.models
=============

用于验证与演示的量子模型（都在 H₀ 本征基下以矩阵形式给出）：

=================  =====================================  ==============================
模型                 H₀ + λV                               解析/基准
=================  =====================================  ==============================
谐振子 + x⁴         ω(n+½) + λ x⁴                          微扰级数渐近发散（无收敛半径）
位移谐振子 −Fx       ω(n+½) − F x                          E_n = ω(n+½) − F²/(2mω²)，
                                                           级数在二阶精确终止
二能级              Δ|1⟩⟨1| + v(|0⟩⟨1|+h.c.)               E_∓ = Δ/2 ∓ √(Δ²/4+v²)
随机非简并矩阵       diag(E_n) + λ·对称随机阵                数值对角化
=================  =====================================  ==============================

所有模型统一封成 :class:`Model`，字段 ``e0`` / ``V`` / ``x`` 可直接喂给
:class:`qmpert.core.PerturbationTheory`，``exact_energy`` 提供解析基准。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

try:                                        # 稀疏支持（可选依赖）
    from scipy import sparse as _sp

    def _issparse(a) -> bool:
        return _sp.issparse(a)

except ImportError:                         # pragma: no cover
    _sp = None

    def _issparse(a) -> bool:
        return False


__all__ = [
    "Model",
    "harmonic_oscillator",
    "anharmonic_oscillator",
    "displaced_oscillator",
    "two_level",
    "random_model",
    "ho_wavefunction",
    "ho_basis_matrix",
    "gaussian_bump",
    "banded_anharmonic",
    "symbolic_anharmonic",
]


# --------------------------------------------------------------------------- #
@dataclass
class Model:
    """一个可直接求解的量子模型。"""

    name: str
    e0: np.ndarray
    V: np.ndarray
    lam_name: str = "λ"
    exact_energy: Optional[Callable[[int, float], float]] = None
    x: Optional[np.ndarray] = None          # 位置算符，用于绘制波函数
    note: str = ""

    def __post_init__(self) -> None:
        self.e0 = np.asarray(self.e0, dtype=float)
        self.V = self.V.tocsr() if _issparse(self.V) else np.asarray(self.V)
        self.sparse = _issparse(self.V)
        if self.x is not None and not _issparse(self.x):
            self.x = np.asarray(self.x, dtype=float)

    # -- 数值精确解（同一截断空间内） ---------------------------------- #
    def hamiltonian(self, lam: float):
        if self.sparse:
            return _sp.diags(self.e0, format="csr") + lam * self.V
        return np.diag(self.e0) + lam * self.V

    def exact_spectrum(self, lam: float = 0.0, k: int = 6) -> np.ndarray:
        """截断空间内的精确谱。

        稠密：全部本征值。稀疏：只算最低的 ``k`` 个（移位反演 + 稀疏 LU），
        返回长度为 k 的升序数组。

        移位量取"最低几个对角元间距"的尺度：太低（例如 Gershgorin 下界）
        会让 ``(H−σ)⁻¹`` 的谱几乎简并，Lanczos 收敛极慢——这在陡峭的
        x⁴ 势上实测会直接不收敛；取在基态附近则几十步即可。
        """
        H = self.hamiltonian(lam)
        if not self.sparse:
            return np.linalg.eigvalsh(H)

        from scipy.sparse.linalg import eigsh

        n = H.shape[0]
        kk = min(int(k), n - 2)
        if kk < 1:
            return np.linalg.eigvalsh(H.toarray())
        d_low = np.sort(H.diagonal())[: min(10, n)]
        gap = float(np.median(np.diff(d_low))) if d_low.size > 1 else 1.0
        sigma = float(d_low[0] - 0.5 * abs(gap) - 1e-12)
        try:
            w = eigsh(H.tocsc(), k=kk, sigma=sigma, which="LM",
                      return_eigenvectors=False, tol=1e-12)
        except Exception:                    # pragma: no cover - 兜底
            w = eigsh(H.tocsc(), k=kk, which="SA", return_eigenvectors=False, tol=1e-10)
        return np.sort(w)

    def exact_state(self, n: int, lam: float) -> np.ndarray:
        """截断空间内第 n 个本征态（会做一次稠密对角化）。"""
        H = self.hamiltonian(lam)
        w, u = np.linalg.eigh(H.toarray() if self.sparse else H)
        return u[:, np.argsort(w)[n]]

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Model {self.name} N={self.e0.size}>"


# --------------------------------------------------------------------------- #
#  谐振子基
# --------------------------------------------------------------------------- #
def harmonic_oscillator(N: int = 200, omega: float = 1.0, mass: float = 1.0):
    """返回 (E_n, x 矩阵)，单位 ℏ=1。"""
    n = np.arange(N)
    e = omega * (n + 0.5)
    pref = np.sqrt(1.0 / (2.0 * mass * omega))
    x = np.zeros((N, N))
    m = np.arange(N - 1)
    x[m, m + 1] = pref * np.sqrt(m + 1.0)
    x[m + 1, m] = x[m, m + 1]
    return e, x


def ho_wavefunction(n: int, xg: np.ndarray, omega: float = 1.0, mass: float = 1.0) -> np.ndarray:
    """谐振子本征函数 ψ_n(x)（ℏ=1），用三项递推实现以避免 H_n(x) 溢出。"""
    xi = np.sqrt(mass * omega) * np.asarray(xg, dtype=float)
    psi0 = (mass * omega / np.pi) ** 0.25 * np.exp(-0.5 * xi**2)
    if n == 0:
        return psi0
    psi1 = np.sqrt(2.0) * xi * psi0
    if n == 1:
        return psi1
    prev, cur = psi0, psi1
    for k in range(1, n):
        prev, cur = cur, np.sqrt(2.0 / (k + 1)) * xi * cur - np.sqrt(k / (k + 1.0)) * prev
    return cur


def ho_basis_matrix(xg: np.ndarray, N: int, omega: float = 1.0, mass: float = 1.0) -> np.ndarray:
    """返回 (N, len(xg)) 的 ψ 矩阵，列为空间点。"""
    psi = np.empty((N, len(xg)))
    psi[0] = ho_wavefunction(0, xg, omega, mass)
    if N > 1:
        psi[1] = ho_wavefunction(1, xg, omega, mass)
    xi = np.sqrt(mass * omega) * np.asarray(xg, dtype=float)
    for k in range(1, N - 1):
        psi[k + 1] = np.sqrt(2.0 / (k + 1)) * xi * psi[k] - np.sqrt(k / (k + 1.0)) * psi[k - 1]
    return psi


def gaussian_bump(N: int = 160, sigma: float = 0.6, L: float = 20.0, M: int = 12000) -> Model:
    """稠密型微扰：V(x) = exp(−x²/2σ²)（偶宇称，在 HO 基下为稠密矩阵）。

    矩阵元由实空间投影 V_kn = ∫ψ_k V ψ_n dx 得到（梯形积分，误差 ~1e-13）。
    用途：演示对称性掩码在**数值生成的稠密矩阵**上剔除禁戒元——奇 Δn 的元
    只是求积噪声（~1e-16），掩码后矩阵严格分块，工作量减半。
    """
    xg = np.linspace(-L, L, M)
    dx = xg[1] - xg[0]
    psi = ho_basis_matrix(xg, N)
    Vx = np.exp(-xg**2 / (2.0 * sigma**2))
    V = (psi * Vx) @ psi.T * dx
    e0 = np.arange(N) + 0.5
    return Model(
        name=f"dense perturbation  V=exp(−x²/2σ²), σ={sigma:g}",
        e0=e0,
        V=V,
        note="稠密矩阵；宇称只允许偶数 Δn，禁戒元为求积噪声。",
    )


# --------------------------------------------------------------------------- #
#  具体模型
# --------------------------------------------------------------------------- #
def anharmonic_oscillator(N: int = 200, power: int = 4, omega: float = 1.0) -> Model:
    """H = p²/2 + ω²x²/2 + λ x^power（power=4 为四次非谐）。"""
    e, x = harmonic_oscillator(N, omega=omega)
    V = np.linalg.matrix_power(x, power)
    return Model(
        name=f"anharmonic ω={omega:g}, V=x^{power}",
        e0=e,
        V=V,
        x=x,
        note="微扰级数为渐近级数：任意阶系数按 m! 增长，收敛半径为零。",
    )


def displaced_oscillator(N: int = 200, omega: float = 1.0, mass: float = 1.0) -> Model:
    """H = H₀ − F x：匀强外场中的谐振子（宇称破缺的奇宇称微扰）。

    精确解：E_n(F) = ω(n+½) − F²/(2mω²)（ℏ=1）。取 m=ω=1 即 E_n − F²/2。
    """
    e, x = harmonic_oscillator(N, omega=omega, mass=mass)
    c = 1.0 / (2.0 * mass * omega**2)
    return Model(
        name="displaced oscillator  H₀ − F x",
        e0=e,
        V=-x,
        lam_name="F",
        x=x,
        exact_energy=lambda n, F: omega * (n + 0.5) - c * F**2,
        note="奇宇称微扰 ⇒ 奇数阶能量修正恒为零；级数在二阶精确终止。",
    )


def two_level(delta: float = 1.0) -> Model:
    """H = Δ|1⟩⟨1| + v(|0⟩⟨1| + |1⟩⟨0|)，解析：E_∓ = Δ/2 ∓ √(Δ²/4+v²)。"""
    e0 = np.array([0.0, delta])
    V = np.array([[0.0, 1.0], [1.0, 0.0]])
    return Model(
        name=f"two-level Δ={delta:g}",
        e0=e0,
        V=V,
        lam_name="v",
        exact_energy=lambda n, v: delta / 2 - np.sqrt(delta**2 / 4 + v**2)
        if n == 0
        else delta / 2 + np.sqrt(delta**2 / 4 + v**2),
        note="E₋ = Δ/2 − Δ/2·√(1+4v²/Δ²)，可展开为 √(1+x) 的幂级数逐阶核对。",
    )


def random_model(N: int = 8, seed: int = 0, spread: float = 1.0) -> Model:
    """随机非简并对角谱 + 随机对称微扰，用作通用正确性测试。"""
    rng = np.random.default_rng(seed)
    gaps = np.sort(rng.uniform(0.5, 1.5, size=N)) * spread
    e0 = np.cumsum(gaps)
    A = rng.normal(size=(N, N))
    V = (A + A.T) / 2
    return Model(name=f"random N={N} seed={seed}", e0=e0, V=V)


# --------------------------------------------------------------------------- #
#  大规模：稀疏非谐振子
# --------------------------------------------------------------------------- #
def banded_anharmonic(N: int = 4000, power: int = 4, omega: float = 1.0) -> Model:
    """同一物理模型（H₀ + λx^power）的稀疏实现，基组可开到 10⁵ 量级。

    x^power 在谐振子基下只有 Δn = 0, ±2, … 的元，带宽 = power。这里用
    升降算符的稀疏矩阵连乘得到，内存 O(power·N)，而不是 O(N²)：

        x = (a + a†)/√2 ,    V = x^power

    对 ``N ≤ 400`` 与 :func:`anharmonic_oscillator` 的稠密结果逐元一致。
    """
    if _sp is None:                         # pragma: no cover
        raise ImportError("banded_anharmonic 需要 scipy")

    idx = np.arange(N)
    off = np.sqrt(idx[1:]) / np.sqrt(2.0)          # ⟨n|x|n+1⟩ = √(n+1)/√2
    X = _sp.diags([off, off], offsets=[1, -1], shape=(N, N), format="csr")

    V = X.copy()
    for _ in range(power - 1):
        V = V @ X
    V = V.tocsr()
    V.eliminate_zeros()

    e0 = omega * (idx + 0.5)
    return Model(
        name=f"sparse anharmonic N={N}, V=x^{power}",
        e0=e0,
        V=V,
        note=f"稀疏 x^{power}：nnz={V.nnz}（稠密需 {N*N}），带宽 {power}。",
    )


def symbolic_anharmonic(N: int = 8, power: int = 4):
    """sympy 精确版非谐振子：``V = x^power`` 由**精确**升降算符构造。

    x 的矩阵元含 √2 等代数数，因此不能用浮点矩阵"四舍五入"回有理数；
    这里直接在 sympy 上做矩阵乘，得到

        x = (a + a†)/√2 ,   V = x^power

    返回 ``(e0, V)``，元素为 ``sympy.Rational`` / ``sympy.sqrt`` 表达式。
    配合 ``PerturbationTheory`` 的 object dtype 后端，可把 E_n⁽ᵐ⁾ 算成
    精确有理数（例：E₀⁽¹⁾=3/4, E₀⁽²⁾=−21/8, E₀⁽³⁾=333/16）。
    """
    try:
        import sympy as sp
    except ImportError:                     # pragma: no cover
        raise ImportError("symbolic_anharmonic 需要 sympy")

    X = sp.zeros(N, N)
    rt2 = sp.sqrt(2)
    for k in range(N - 1):
        val = sp.sqrt(k + 1) / rt2
        X[k, k + 1] = val
        X[k + 1, k] = val

    V = X**power
    e0 = [sp.Rational(2 * n + 1, 2) for n in range(N)]
    return e0, V
