"""讲义第 12 章 §4 例 1：二能级系统  H = (δ/2)σ_z + (Ω/2)σ_x  （ħ = 1）。

讲义式 (44)–(47) 给出
    E_0 = -(δ/2) - Ω²/(4δ) + O(Ω⁴)
    E_1 = +(δ/2) + Ω²/(4δ) + O(Ω⁴)
    |0> = |↓>(1 - Ω²/(8δ²)) - |↑>(Ω/2δ) + O(Ω³)

程序约定 H = H₀ + λV，λ 就是讲义里的 Ω。所以 V 要取“Ω = 1”时的矩阵：
    e0 = [-δ/2, +δ/2]             H₀ 的两个对角元
    V  = [[0, 1/2], [1/2, 0]]     (ħ/2)σ_x

运行：
    python examples/ex1_two_level.py
"""

import argparse
from fractions import Fraction
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 免安装直接运行
from qmpert import ParitySymmetry, PerturbationTheory

PROJECT = Path(__file__).resolve().parents[1]
DELTA = Fraction(1)

# object 后端让全部递推保持有理数运算，不把 0.1、1/3 等先换成二进制浮点。
e0 = np.array([-DELTA / 2, DELTA / 2], dtype=object)
V = np.array(
    [[Fraction(0), Fraction(1, 2)], [Fraction(1, 2), Fraction(0)]], dtype=object
)

# V 只连接 |↑> 与 |↓>：在以 σ_z 为对角基的表象里它是“奇”算符，奇数阶修正必为零。
sym = ParitySymmetry([+1, -1], perturbation_parity=-1)
pt = PerturbationTheory(e0, V, symmetry=sym, name="two-level")


def _evaluate_coeffs(coeffs: np.ndarray, lam: Fraction, order: int) -> np.ndarray:
    """精确代入 λ；不经过 normalized_state 的浮点范数。"""
    return sum((lam**m * coeffs[m] for m in range(order + 1)), np.zeros(2, dtype=object))


def _make_figure(s0, output: Path) -> None:
    """能量、截断误差和自旋混合概率，均与解析解对照。"""
    from qmpert.viz import COL, use_style
    import matplotlib.pyplot as plt

    use_style()
    ratios = np.linspace(0.0, 0.85, 150)
    exact_e = -0.5 * np.hypot(1.0, ratios)
    fig, (ax_e, ax_err, ax_p) = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    ax_e.plot(ratios, exact_e, color=COL["ink"], lw=2.6, label="Analytic")
    for order, color in zip((2, 4, 6), (COL["blue"], COL["red"], COL["teal"])):
        energies = s0.energy(ratios, order).astype(float)
        error = abs(energies - exact_e)
        mask = error > 0
        ax_e.plot(ratios, energies, "--", color=color, label=f"PT({order})")
        ax_err.semilogy(ratios[mask], error[mask], color=color, label=f"PT({order})")
    ax_e.set(xlabel=r"$\Omega/\delta$", ylabel=r"$E_-/\delta$", title="Ground-state energy")
    ax_err.set(xlabel=r"$\Omega/\delta$", ylabel=r"$|E_{\rm PT}-E_{\rm exact}|/\delta$",
               title="Energy truncation error")
    ax_e.legend()
    ax_err.legend()

    unit = np.asarray(s0.normalized_coeffs()[:3], dtype=float)
    vec = np.polynomial.polynomial.polyval(ratios, unit)
    p_pt = abs(vec[1]) ** 2 / np.sum(abs(vec) ** 2, axis=0)
    p_exact = (1 - 1 / np.hypot(1.0, ratios)) / 2
    ax_p.plot(ratios, p_exact, color=COL["ink"], lw=2.6, label="Analytic")
    ax_p.plot(ratios, p_pt, "--", color=COL["blue"], label="Normalized PT(2)")
    ax_p.set(xlabel=r"$\Omega/\delta$", ylabel=r"$|\langle\uparrow|\psi_-\rangle|^2$", title="State mixing")
    ax_p.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="二能级非简并微扰：精确系数、归一化态矢与图")
    ap.add_argument("--output", type=Path, default=PROJECT / "figures" / "ex1_two_level.png")
    args = ap.parse_args()

    s0 = pt.solve(state=0, max_order=6)
    s1 = pt.solve(state=1, max_order=6)
    print("二能级（ℏ = δ = 1，λ = Ω），基序 |↓>, |↑>")
    print(f"选择定则自洽性：禁戒元最大绝对值 = {pt.violation:.1e}")
    print(f"先验预测的零阶 = {sorted(s0.zeros_predicted)}，实测 = {sorted(s0.measured_zero_orders())}")

    analytic = {2: -1 / (4 * DELTA), 4: 1 / (16 * DELTA**3), 6: -1 / (32 * DELTA**5)}
    print("\n  m    E₋⁽ᵐ⁾      E₊⁽ᵐ⁾     E₋⁽ᵐ⁾ 解析值")
    for m in range(1, 7):
        print(f"  {m}    {s0.dE[m]!s:>8}    {s1.dE[m]!s:>8}    {analytic.get(m, Fraction(0))!s:>8}")

    unit = s0.normalized_coeffs()
    print("\n范数归一化的逐阶态矢系数（基序 |↓>, |↑>）：")
    for m in range(3):
        print(f"  |psi^({m})> -> [{unit[m, 0]}, {unit[m, 1]}]")
    print("  对照：|psi^(2)> 的 |↓> 分量 = -1/8，即讲义式 (46) 的归一化项。")

    omega = Fraction(1, 5)
    intermediate = _evaluate_coeffs(s0.coeffs, omega, 2)
    normalized = _evaluate_coeffs(unit, omega, 2)
    print(f"\nΩ = {omega} 时，截断到二阶：")
    print(f"  中间归一化 : [{intermediate[0]}, {intermediate[1]}]")
    print(f"  范数归一化 : [{normalized[0]}, {normalized[1]}]  （范数平方误差为 O(Ω⁴)）")

    print("\nΩ/δ     6 阶截断能量        精确解              差值")
    for ratio in (0.05, 0.1, 0.2, 0.5, 1.0):
        exact = -0.5 * np.sqrt(1.0 + ratio**2)
        val = float(s0.energy(Fraction(str(ratio)), 6))
        print(f"{ratio:5.2f}   {val: .14f}   {exact: .14f}   {val - exact:+.2e}")

    _make_figure(s0, args.output)
    print(f"\n图已保存：{args.output}")
    print("注：真正的小参数是 Ω/δ；Ω/δ 较大时应看严格对角化，不要盲目增加截断阶。")


if __name__ == "__main__":
    main()
