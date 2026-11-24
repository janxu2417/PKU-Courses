"""讲义第 12 章 §7.1：二阶简并微扰的三能级例题（式 110–124）。

取 H = E0 [diag(0, 1, 1) + λ V]，λ = V0/E0，V 为式 (111) 无量纲矩阵。
程序输出精确好基、(E^(1), E^(2)) 和一阶态矢，并绘制二阶能量与严格对角化的对照。
运行：python examples/ex3_three_level_degenerate.py
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmpert.degenerate import DegeneratePerturbationTheory

PROJECT = Path(__file__).resolve().parents[1]


def exact_model():
    rt2 = sp.sqrt(2)
    e0 = np.array([0, 1, 1], dtype=object)
    V = np.array(
        [[0, rt2, 2 * rt2], [rt2, 0, 2], [2 * rt2, 2, 0]], dtype=object
    )
    return e0, V


def _vec_text(vec) -> str:
    return "[" + ", ".join(str(sp.simplify(x)) for x in vec) + "]"


def _make_figure(e0, V, result, output: Path) -> None:
    from qmpert.viz import COL, use_style
    import matplotlib.pyplot as plt

    use_style()
    e0f = np.asarray(e0, dtype=float)
    Vf = np.asarray(V, dtype=float)
    lams = np.geomspace(0.001, 0.12, 120)
    exact = np.array([np.linalg.eigvalsh(np.diag(e0f) + x * Vf) for x in lams])
    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    colors = [COL["blue"], COL["red"], COL["teal"]]
    labels = [r"$E_{11}$", r"$E_{21}$", r"$E_{22}$"]
    for j, (branch, color, label) in enumerate(zip(result.branches, colors, labels)):
        predicted = np.array([float(branch.energy(x)) for x in lams])
        ax.plot(np.r_[0, lams], np.r_[float(branch.e0), exact[:, j]], color=color, lw=2.6)
        ax.plot(np.r_[0, lams], np.r_[float(branch.e0), predicted], "--", color=COL["ink"], lw=1.4)
        markers = {"ls": "none", "marker": "o", "markevery": 12, "ms": 4.0} if j == 0 else {}
        ax_err.loglog(lams, abs(predicted - exact[:, j]), color=color, label=label,
                      zorder=4 if j == 0 else 3, **markers)
    ax_err.loglog(lams, 10 * lams**3, ":", color=COL["gray"], label=r"$10\lambda^3$")
    ax.set(xlabel=r"$\lambda=V_0/E_0$", ylabel=r"$E/E_0$",
           title="Diagonalization (solid) and PT(2) (dashed)")
    ax_err.set(xlabel=r"$\lambda$", ylabel=r"$|E_{\rm PT(2)}-E_{\rm diag}|/E_0$",
               title="Second-order energy error")
    ax_err.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="三能级二阶简并微扰：精确讲义对照与能级图")
    ap.add_argument("--output", type=Path,
                    default=PROJECT / "figures" / "ex3_three_level_degenerate.png")
    args = ap.parse_args()

    e0, V = exact_model()
    result = DegeneratePerturbationTheory(e0, V).solve()
    labels = ["|1,1>", "|2,1>（反对称）", "|2,2>（对称）"]
    print("三能级简并微扰（λ = V₀/E₀，能量以 E₀ 为单位）")
    print("系数的基序为 |1>, |2>, |3>。")
    for label, branch in zip(labels, result.branches):
        print(f"\n{label}  E⁽⁰⁾ = {branch.e0}，(E1, E2) = ({branch.dE1}, {branch.dE2})")
        print("  好基 c0       =", _vec_text(branch.c0))
        print("  一阶修正 c1   =", _vec_text(branch.c1))
        print("  |psi> = c0 + λ c1 + O(λ^2)，范数条件在一阶成立")

    print("\n能量对照（E/E0）：")
    print("  E11 = -10 λ^2 + O(λ^3)")
    print("  E21 = 1 - 2 λ + λ^2 + O(λ^3)")
    print("  E22 = 1 + 2 λ + 9 λ^2 + O(λ^3)")
    print("  对应讲义式 (119)–(121)：二阶系数 (-10, 1, 9)。")
    _make_figure(e0, V, result, args.output)
    print(f"\n图已保存：{args.output}")


if __name__ == "__main__":
    main()
