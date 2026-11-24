"""讲义第 12 章 §4 例 2：无限深方势阱中心的 δ 函数势。

    H₀ = P²/2M + U(X)，  U = 0 (0 < x < L)，其余为 ∞
    V  = (ħ²/ML) δ(X - L/2)，  取 ħ = M = L = 1

基矢 φ_n(x) = √(2/L) sin(nπx/L)，n = 1,2,3,…，E_n⁽⁰⁾ = π²n²/2。
在这个基下（讲义式 57、58）

    V_mn = 2 sin(mπ/2) sin(nπ/2) = 2 s_m s_n ,     s_n = sin(nπ/2)

即 V = 2 s sᵀ，秩 1。偶数 n 的 s_n = 0：偶数态在 x = L/2 有节点，与杂质完全解耦。

运行：
    python examples/ex2_delta_well.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 免安装直接运行
from qmpert import ParitySymmetry, PerturbationTheory

PROJECT = Path(__file__).resolve().parents[1]
SECOND_ORDER_LIMIT = -2 / np.pi**2


def exact_energy(lam: float) -> float:
    """由中心的导数跳跃条件求无穷基组基态能量。"""
    if lam == 0:
        return np.pi**2 / 2
    if lam == -2:
        return 0.0
    if lam < -2:
        def equation(kappa):
            return (2.0 if kappa == 0 else kappa / np.tanh(kappa / 2)) + lam

        return -brentq(equation, 0.0, -lam) ** 2 / 2

    def equation(k):
        return (2.0 if k == 0 else k / np.tan(k / 2)) + lam

    bounds = (np.pi, np.nextafter(2 * np.pi, 0.0)) if lam > 0 else (0.0, np.pi)
    return brentq(equation, *bounds) ** 2 / 2


def finite_basis_energy(e0: np.ndarray, lam: float) -> float:
    """秩 1 世俗方程给出同一有限基组的严格本征值。"""
    if lam == 0:
        return e0[0]
    gaps = e0[2::2] - e0[0]

    def equation(shift):
        return shift * (1 + 2 * lam * np.sum(1 / (gaps - shift))) - 2 * lam

    if lam > 0:
        bounds = (0.0, min(2 * lam, np.nextafter(gaps[0], 0.0)))
    else:
        bounds = (2 * lam * (len(gaps) + 1), 0.0)
    return e0[0] + brentq(equation, *bounds)


def sine_series(x: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    z = np.exp(1j * np.pi * x)
    return np.sqrt(2) * np.imag(z * np.polynomial.polynomial.polyval(z, coeffs))


def make_figure(series, sizes, errors, output: Path) -> None:
    from qmpert.viz import COL, use_style
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, StrMethodFormatter

    use_style()
    fig, (ax_wave, ax_error) = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    x = np.linspace(0, 1, 1601)
    analytic = np.sqrt(2) * (
        -np.sin(np.pi * x) / np.pi**2 + 2 / np.pi * (x - (x >= 0.5)) * np.cos(np.pi * x)
    )
    ax_wave.plot(x, sine_series(x, series.coeffs[1]), color=COL["blue"], lw=2.6,
                 label=f"Basis sum, $N={series.e0.size}$")
    ax_wave.plot(x, analytic, "--", color=COL["ink"], label="Analytic, Eq. (60)")
    ax_wave.axvline(0.5, color=COL["gray"], lw=0.9)
    ax_wave.set(xlabel=r"$x/L$", ylabel=r"$\sqrt{L}\,\psi_1^{(1)}(x)$",
                title="First-order wavefunction")
    ax_wave.legend()
    ax_error.loglog(sizes, errors, "o-", color=COL["teal"], label="Basis truncation")
    ax_error.loglog(sizes, errors[0] * sizes[0] / sizes, "--", color=COL["ink"], label=r"$1/N$")
    ax_error.set_xticks(sizes)
    ax_error.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax_error.xaxis.set_minor_formatter(NullFormatter())
    ax_error.set(xlabel="Basis dimension $N$", ylabel=r"$|E_{1,N}^{(2)}-E_{1,\infty}^{(2)}|$",
                 title="Second-order coefficient error")
    ax_error.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="中心 δ 势阱的微扰系数、波函数与基组收敛")
    ap.add_argument("--basis", type=int, default=2000, help="基组维数，至少 16")
    ap.add_argument("--output", type=Path, default=PROJECT / "figures" / "ex2_delta_well.png")
    args = ap.parse_args()
    if args.basis < 16:
        ap.error("--basis 至少为 16")
    N = args.basis
    n = np.arange(1, N + 1)
    e0 = (np.pi * n) ** 2 / 2
    s = np.where(n % 2 == 1, (-1.0) ** ((n - 1) // 2), 0.0)
    par = np.where(n % 2 == 1, 1, -1)
    engine = PerturbationTheory(e0, 2 * np.outer(s, s), symmetry=ParitySymmetry(par, 1),
                                enforce_symmetry=False)
    s1, s2, s3 = engine.solve_many((0, 1, 2), 6)

    print(f"中心 δ 势阱（ℏ = M = L = 1，N = {N}），量子数 n 从 1 起")
    print(f"宇称禁戒元最大绝对值 = {engine.violation:.1e}")
    print("\n n     E⁽¹⁾        有限基 E⁽²⁾         讲义式 (62)、(64)")
    for quantum, series in enumerate((s1, s2, s3), 1):
        analytic = SECOND_ORDER_LIMIT / quantum**2 if quantum % 2 else 0.0
        print(f"{quantum:2d}   {series.dE[1]:6.2f}   {series.dE[2]: 17.12f}   {analytic: 17.12f}")
    print("偶数态 n=2：第 1–6 阶均为零，中心节点使全部耦合矩阵元为零。")

    # 奇数 n 高能尾项的首项与 1/N 成正比。少量 N 采用闭式对角求和。
    sizes = np.array(sorted({N // 8, N // 4, N // 2, N, 2 * N}))
    corrections = np.array([
        -8 / np.pi**2 * np.sum(1 / (np.arange(3, size + 1, 2, dtype=float)**2 - 1))
        for size in sizes
    ])
    errors = np.abs(corrections - SECOND_ORDER_LIMIT)
    extrapolated = 2 * corrections[-1] - corrections[-2]
    print("\n基态二阶系数的基组收敛")
    for size, coefficient, error in zip(sizes, corrections, errors):
        print(f"  N = {size:5d}   E⁽²⁾ = {coefficient: .12f}   |误差| = {error:.3e}")
    print(f"  外推 2 E⁽²⁾(2N) − E⁽²⁾(N) = {extrapolated:.12f}，"
          f"|误差| = {abs(extrapolated - SECOND_ORDER_LIMIT):.3e}")

    print("\n    λ        PT(2)         PT(6)      无限基准      基组误差      阶数误差(6)")
    for lam in (-0.3, -0.1, 0.05, 0.1, 0.2):
        finite, exact = finite_basis_energy(e0, lam), exact_energy(lam)
        pt2, pt6 = s1.energy(lam, 2), s1.energy(lam, 6)
        print(f"{lam:5.2f}  {pt2:13.9f} {pt6:13.9f} {exact:13.9f}"
              f"  {finite - exact:+.2e}    {pt6 - finite:+.2e}")
    print("  基组误差 = 有限基严格解 − 无限基准。阶数误差 = PT(6) − 有限基严格解。")

    c3 = s1.coeffs[1, 2]
    cusp = sine_series(np.array([0.5]), s1.coeffs[1])[0]
    print(f"\n一阶波函数：c₃⁽¹⁾ = {c3:.12f}（式 (59)：{1 / (2 * np.pi**2):.12f}）")
    print(f"  ψ₁⁽¹⁾(L/2) = {cusp:.10f}（式 (60)：{-np.sqrt(2) / np.pi**2:.10f}）")
    make_figure(s1, sizes, errors, args.output)
    print(f"\n图已保存：{args.output}")


if __name__ == "__main__":
    main()
