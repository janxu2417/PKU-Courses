"""
qmpert.cli
==========

命令行入口：

    python -m qmpert --model x4 --N 200 --state 0 --order 12 --lam 0.05
    python -m qmpert --model x4 --order 20 --lam 0.2 --resum
    python -m qmpert --model x4 --N 20000 --sparse --order 40

设计目标是"一条命令看到该看的数字"：前几阶系数、给定 λ 下的截断能量、
精确值（若模型有解析解或可对角化）与相对误差；``--resum`` 追加 Padé /
Borel–Padé 的估计。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _build(args):
    from .models import (
        Model,
        anharmonic_oscillator,
        banded_anharmonic,
        displaced_oscillator,
        random_model,
    )

    if args.model == "x4":
        if args.sparse:
            return banded_anharmonic(N=args.N)
        return anharmonic_oscillator(N=args.N)
    if args.model == "displaced":
        return displaced_oscillator(N=args.N)
    if args.model == "two-level":
        # 与讲义式 (41)–(47) 及 --exact 完全同约定：H=δS_z+ΩS_x，δ=ħ=1。
        return Model(
            name="two-level (lecture convention, δ=1)",
            e0=np.array([-0.5, 0.5]),
            V=0.5 * np.array([[0.0, 1.0], [1.0, 0.0]]),
            lam_name="Ω",
            exact_energy=lambda n, omega: (0.5 if n else -0.5) * np.sqrt(1 + omega**2),
        )
    if args.model == "random":
        return random_model(N=max(args.N, 4), seed=args.seed)
    raise ValueError(args.model)


def _exact_operators(args, model):
    """返回保持 object dtype 的精确算符；数值 model 仍用于名称等元数据。"""
    from fractions import Fraction

    if args.model == "two-level":
        half = Fraction(1, 2)
        e0 = np.array([-half, half], dtype=object)
        V = np.array([[0, half], [half, 0]], dtype=object)
        return e0, V
    if args.model == "x4":
        from .models import symbolic_anharmonic

        e0, V = symbolic_anharmonic(N=args.N)
        return np.array(e0, dtype=object), np.array(V.tolist(), dtype=object)
    raise ValueError("--exact 仅支持 two-level 和 x4 模型")


def _print_components(vec, exact: bool, limit: int | None = None) -> None:
    indices = [i for i, value in enumerate(vec) if value != 0]
    omitted = 0
    if limit is not None and len(indices) > limit:
        ranked = sorted(indices, key=lambda i: float(abs(vec[i])), reverse=True)
        indices = sorted(ranked[:limit])
        omitted = len(ranked) - limit
    if not indices:
        print("    （零向量）")
        return
    for i in indices:
        value = str(vec[i]) if exact else f"{vec[i]:.10g}"
        print(f"    c[{i}] = {value}")
    if omitted:
        print(f"    …（省略其余 {omitted} 个分量）")


def _exact_normalized_state(series, lam: float, order: int):
    """把命令行给定的十进制耦合转为有理数后归一化。"""
    import sympy as sp

    return series.normalized_state(sp.Rational(str(lam)), order)


def _show_state(series, args) -> None:
    print()
    print(f"逐阶范数归一化修正（展示第 0–{min(args.order, 2)} 阶的非零分量）：")
    corrections = series.normalized_coeffs()
    for m in range(min(args.order, 2) + 1):
        print(f"  m={m}")
        _print_components(corrections[m], args.exact)

    print(f"截断并整体归一化的态矢（λ={args.lam:g}，m={args.order}）：")
    if args.exact:
        state = _exact_normalized_state(series, args.lam, args.order)
    else:
        state = series.normalized_state(args.lam, args.order)
    _print_components(state, args.exact, limit=8)


def main(argv: list[str] | None = None) -> int:
    from .core import PerturbationTheory

    ap = argparse.ArgumentParser(
        prog="qmpert",
        description="任意阶非简并 Rayleigh–Schrödinger 微扰论",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default="x4", choices=["x4", "displaced", "two-level", "random"])
    ap.add_argument("--N", type=int, default=200, help="基组维数")
    ap.add_argument("--state", type=int, default=0, help="目标态指标")
    ap.add_argument("--order", type=int, default=12, help="展开阶数")
    ap.add_argument("--lam", type=float, default=0.05, help="耦合强度")
    ap.add_argument("--sparse", action="store_true", help="用稀疏后端（仅 x4）")
    ap.add_argument("--resum", action="store_true", help="追加 Padé / Borel–Padé 估计")
    ap.add_argument("--seed", type=int, default=0, help="random 模型的随机种子")
    ap.add_argument("--exact", action="store_true", help="用精确有理算术（仅 two-level / x4）")
    ap.add_argument("--show-state", action="store_true", help="显示逐阶归一化修正与截断态矢")
    ap.add_argument("--plot", metavar="PATH", help="保存能量阶数与态概率的 1×2 PNG 图")
    args = ap.parse_args(argv)

    if args.exact and args.model not in {"two-level", "x4"}:
        ap.error("--exact 仅支持 two-level 和 x4 模型")
    if args.exact and args.sparse:
        ap.error("--exact 使用稠密符号算符，不能与 --sparse 同用")

    model = _build(args)
    e0, V = _exact_operators(args, model) if args.exact else (model.e0, model.V)
    engine = PerturbationTheory(e0, V, name=model.name)
    series = engine.solve(args.state, args.order)

    n = args.state
    print(f"模型      : {model.name}")
    print(f"基组      : N = {e0.size}" + ("（稀疏）" if getattr(model, "sparse", False) and not args.exact else ""))
    e0_text = str(e0[n]) if args.exact else f"{series.unperturbed_energy:.10g}"
    print(f"目标态    : n = {n}，E⁰ = {e0_text}")
    print(f"展开阶数  : {args.order}，耦合 λ = {args.lam:g}")
    if series.zeros_predicted:
        print(f"对称性    : {series.symmetry_name}，预测零阶 {sorted(series.zeros_predicted)}")
    print()
    print(f"{'m':>3}  {'E⁽ᵐ⁾':>20}")
    for m in range(1, min(args.order, 12) + 1):
        if args.exact:
            print(f"{m:>3}  {str(series.dE[m]):>20}")
        else:
            print(f"{m:>3}  {series.dE[m]:>20.10e}")
    if args.order > 12:
        print(f"…     （共 {args.order} 阶，前 12 阶见上）")

    if args.exact:
        from fractions import Fraction

        e_pt = series.energy_exact(Fraction(str(args.lam)))
    else:
        e_pt = series.energy(args.lam, args.order)
    e_pt_float = float(np.real(e_pt))
    print()
    energy_text = str(e_pt) if args.exact else f"{e_pt_float:.14f}"
    print(f"截断能量 E(λ, m={args.order}) = {energy_text}")
    # 严格对角化求同一有限基组的数值基准。
    try:
        if args.exact:
            H = np.diag(np.asarray(e0, dtype=float)) + args.lam * np.asarray(V, dtype=float)
            e_exact = float(np.linalg.eigvalsh(H)[n])
        else:
            e_exact = float(model.exact_spectrum(args.lam, k=n + 1)[n])
        print(f"对角化基准 E_diag       = {e_exact:.14f}")
        print(f"误差     |ΔE|           = {abs(e_pt_float - e_exact):.3e}")
    except Exception as exc:  # pragma: no cover - 仅影响提示信息
        print(f"（未计算精确值：{exc}）")

    if args.show_state:
        _show_state(series, args)

    if args.plot:
        from .viz import fig_series_overview

        fig_series_overview(series, args.lam, args.order, args.plot)
        print(f"图已保存 : {args.plot}")

    if args.resum:
        from .resum import BorelPade, PadeTable

        coeffs = np.concatenate(([series.unperturbed_energy], np.asarray(series.dE[1:], float)))
        print()
        try:
            tab = PadeTable(coeffs, max_order=min(12, len(coeffs) // 2 - 1))
            val, k, delta = tab.best(args.lam)
            print(f"Padé      [{k}/{k}]        = {val:.14f}   (|Δ| = {delta:.1e})")
        except Exception as exc:
            print(f"Padé      失败：{exc}")
        try:
            bp = BorelPade(coeffs, min(16, len(coeffs) // 2 - 1))
            print(f"Borel–Padé                = {float(bp(args.lam)):.14f}")
        except Exception as exc:
            print(f"Borel–Padé 不可用：{exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
