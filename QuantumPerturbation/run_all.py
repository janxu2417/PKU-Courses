"""
run_all.py —— 一键跑完全部结果：求解 → 绘图 → 生成 HTML 报告

    python3 run_all.py

产物
----
figures/*.png    八张可视化图
results.json     关键数值（供报告与复现）
report.html      自包含 HTML 报告（图片内嵌 base64）
"""

from __future__ import annotations

import os

# 220×220 的 eigh 在默认 BLAS 线程下会被超订拖慢 500 倍，必须在 import numpy 前限制
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import json
import time
from pathlib import Path

import numpy as np

from qmpert import (
    ParitySymmetry,
    DeltaSymmetry,
    PerturbationTheory,
    anharmonic_oscillator,
    displaced_oscillator,
    gaussian_bump,
)
from qmpert import report, viz

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

N_BASIS = 220
MAX_ORDER = 40
STATES = (0, 1, 2, 3, 4)


def main() -> dict:
    t0 = time.perf_counter()
    summary: dict = {}

    # ------------------------------------------------------------------ #
    # A. 四次非谐振子：任意阶系数、收敛性、渐近性
    # ------------------------------------------------------------------ #
    print("[1/9] 四次非谐振子 x^4 ...")
    m4 = anharmonic_oscillator(N=N_BASIS, power=4)
    pt4 = PerturbationTheory(m4.e0, m4.V, name=m4.name)
    series4 = {n: pt4.solve(n, MAX_ORDER) for n in STATES}

    print(f"{'m':>3}" + "".join(f"{f'E{n}^(m)':>14}" for n in STATES))
    for k in range(1, 9):
        print(f"{k:3d}" + "".join(f"{series4[n].dE[k]:14.6g}" for n in STATES))

    viz.fig_convergence(m4, series4[0], n=0, lam_max=0.04, orders=(0, 1, 2, 3, 4),
                        path=str(FIG / "fig1_convergence.png"))

    lams_asy = (0.015, 0.02, 0.05)
    viz.fig_asymptotic(m4, series4, lams=lams_asy, max_order=MAX_ORDER,
                       path=str(FIG / "fig2_asymptotic.png"))

    asy = {}
    for lam in lams_asy:
        ex = m4.exact_spectrum(lam)[0]
        errs = np.array([abs(series4[0].energy(lam, k) - ex) for k in range(1, MAX_ORDER + 1)])
        k = int(np.argmin(errs)) + 1
        asy[f"{lam:g}"] = {
            "m_star": k,
            "err_min": float(errs[k - 1]),
            "err_first": float(errs[0]),
            "err_last": float(errs[-1]),
            "rel_min": float(errs[k - 1] / abs(ex)),
        }
    summary["asymptotic"] = asy

    # 与解析一、二阶系数对照
    ana = {}
    for n in range(4):
        ana[str(n)] = {
            "E1_num": float(series4[n].dE[1]),
            "E1_ana": 0.75 * (2 * n**2 + 2 * n + 1),
            "E2_num": float(series4[n].dE[2]),
            "E2_ana": -(34 * n**3 + 51 * n**2 + 59 * n + 21) / 8.0,
        }
    summary["analytic_check"] = ana
    summary["coeffs"] = {str(n): [float(x) for x in series4[n].dE[1:9]] for n in STATES}
    summary["model"] = {"N": N_BASIS, "max_order": MAX_ORDER}

    # ------------------------------------------------------------------ #
    # B. 对称性：位移谐振子（奇宇称微扰）
    # ------------------------------------------------------------------ #
    print("[2/9] 位移谐振子（宇称选择定则）...")
    md = displaced_oscillator(N=200)
    sym_odd = ParitySymmetry((-1) ** np.arange(md.e0.size), perturbation_parity=-1)
    ptd = PerturbationTheory(md.e0, md.V, symmetry=sym_odd, name=md.name)
    sd = ptd.solve(0, 10)
    viz.fig_symmetry(md, sd, F_max=1.6, path=str(FIG / "fig3_symmetry.png"))

    summary["stark"] = {
        "violation": sym_odd.validate(md.V),
        "predicted": sorted(sym_odd.predicted_zero_orders(0, 10)),
        "measured": sorted(sd.measured_zero_orders()),
        "E2": float(sd.dE[2]),
        "max_odd": float(max(abs(sd.dE[k]) for k in range(1, 11, 2))),
        "max_ge3": float(max(abs(sd.dE[k]) for k in range(3, 11))),
        "exact_check": {
            f"{F:g}": float(sd.energy(F, 2) - md.exact_energy(0, F)) for F in (0.2, 0.5, 1.0, 1.5)
        },
    }

    # ------------------------------------------------------------------ #
    # C. 稀疏结构：带状 x^4 vs 稠密凸起
    # ------------------------------------------------------------------ #
    print("[3/9] 选择定则与矩阵稀疏结构 ...")
    band_sym = DeltaSymmetry({0, 2, -2, 4, -4}, name=r"Δn ∈ {0,±2,±4}")
    mg = gaussian_bump(N=140)
    sym_even = ParitySymmetry((-1) ** np.arange(mg.e0.size), perturbation_parity=+1)
    ptg = PerturbationTheory(mg.e0, mg.V, symmetry=sym_even, enforce_symmetry=True)
    viz.fig_sparsity(m4, mg, sym_even, path=str(FIG / "fig4_sparsity.png"))
    d = ptg.diagnostics()
    summary["sparsity"] = {
        "band_nnz_frac": float(np.count_nonzero(m4.V) / m4.V.size),
        "violation": band_sym.validate(m4.V),
        "dense_violation": sym_even.validate(mg.V),
        "dense_nnz_ratio": d["nnz_ratio"],
        "dense_N": int(mg.e0.size),
        "band_rule": band_sym.summary(),
    }

    # ------------------------------------------------------------------ #
    # D. 波函数修正
    # ------------------------------------------------------------------ #
    print("[4/9] 波函数修正 ...")
    lam_w = 0.05
    xg = np.linspace(-6.0, 6.0, 720)
    l2 = viz.fig_wavefunction(m4, series4[0], lam_w, 0, xg, orders=(0, 1, 2, 3),
                              path=str(FIG / "fig5_wavefunction.png"))
    summary["wavefunction"] = {
        "lam": lam_w,
        "l2": [float(v) for v in l2],
        "orders": list(range(len(l2))),
    }

    # ------------------------------------------------------------------ #
    # E. 能级图
    # ------------------------------------------------------------------ #
    print("[5/9] 能级演化 ...")
    viz.fig_levels(m4, [series4[n] for n in STATES[:4]], lam_max=0.025, orders=(1, 3),
                   path=str(FIG / "fig6_levels.png"))

    # ------------------------------------------------------------------ #
    # F. 精确算术：把前几阶系数算成有理数
    # ------------------------------------------------------------------ #
    print("[6/9] 精确算术（sympy 后端）...")
    import sympy as sp

    from qmpert.models import symbolic_anharmonic

    e0s, Vs = symbolic_anharmonic(N=20)
    pt_sym = PerturbationTheory(np.array(e0s, dtype=object),
                                np.array(Vs.tolist(), dtype=object))
    s_sym = pt_sym.solve(0, 6)
    exact = {
        "coeffs": [str(sp.nsimplify(x)) for x in s_sym.dE[1:7]],
        "float": [float(x) for x in s_sym.dE[1:7]],
        "N": 20,
    }
    summary["exact"] = exact
    print("      ", " ".join(exact["coeffs"]))

    # ------------------------------------------------------------------ #
    # G. 重求和：把发散级数变成可用结果
    # ------------------------------------------------------------------ #
    print("[7/9] 重求和（Padé / Borel–Padé）...")
    from qmpert.resum import BorelPade, PadeTable

    s0 = series4[0]
    coeffs0 = np.concatenate(([s0.unperturbed_energy], np.asarray(s0.dE[1:], float)))
    viz.fig_resummation(m4, s0, coeffs0, lam_order=0.05,
                        path=str(FIG / "fig7_resummation.png"))

    tab = PadeTable(coeffs0, max_order=20)
    bp = BorelPade(coeffs0, 16)
    bp20 = BorelPade(coeffs0, 20)
    poles20 = bp20.dangerous_poles(0.2)
    res: dict = {
        "pade_residual": float(tab.approx[20].residual_mp),
        # [20/20] 为何不可用：交给报告渲染，避免模板里写死极点位置
        "borel20": {
            "order": 20,
            "lam": 0.2,
            "safe": bool(bp20.is_safe(0.2)),
            "poles_t": [float(np.real(z)) for z in poles20],
        },
    }
    for lam in (0.05, 0.1, 0.2, 0.5, 1.0):
        ex = m4.exact_spectrum(lam)[0]
        errs = np.array([abs(s0.energy(lam, k) - ex) for k in range(1, MAX_ORDER + 1)])
        k = int(np.argmin(errs)) + 1
        pv, pk, _ = tab.best(lam)
        try:
            bv, bk = float(bp(lam)), 16
        except ValueError:
            bv, bk = float("nan"), -1
        res[f"{lam:g}"] = {
            "exact": float(ex),
            "trunc_best": float(errs[k - 1]),
            "trunc_order": k,
            "trunc_20": float(errs[19]),
            "pade": float(abs(pv - ex)),
            "pade_order": int(pk),
            "borel": float(abs(bv - ex)),
            "borel_order": int(bk),
        }
    summary["resum"] = res

    # ------------------------------------------------------------------ #
    # H. 规模：同一个模型，稀疏 vs 稠密
    # ------------------------------------------------------------------ #
    print("[8/9] 规模扩展（稀疏 vs 稠密）...")
    from qmpert.models import banded_anharmonic

    def _bench(N: int, dense: bool):
        mm = banded_anharmonic(N=N)
        Vm = mm.V.toarray() if dense else mm.V
        engine = PerturbationTheory(mm.e0, Vm)
        best = float("inf")
        for _ in range(3):
            t1 = time.perf_counter()
            ser = engine.solve(0, 40)
            best = min(best, time.perf_counter() - t1)
        return best, engine.diagnostics(), float(ser.energy(0.02, 20))

    scal = {
        "dense": {"N": [], "sec": [], "mb": []},
        "sparse": {"N": [], "sec": [], "mb": [], "nnz": []},
        "energy": {},
        "lambda": 0.02,
        "orders": 40,
    }
    for N in (100, 200, 400, 800, 1600, 3200):
        dt, dg, _ = _bench(N, True)
        scal["dense"]["N"].append(N)
        scal["dense"]["sec"].append(dt)
        scal["dense"]["mb"].append(dg["dense_mb"])
    for N in (100, 400, 1600, 3200, 6400, 25600, 102400):
        dt, dg, en = _bench(N, False)
        scal["sparse"]["N"].append(N)
        scal["sparse"]["sec"].append(dt)
        scal["sparse"]["mb"].append(dg["stored_mb"])
        scal["sparse"]["nnz"].append(dg["nnz_used"])
        scal["energy"][str(N)] = en
    i3200 = scal["sparse"]["N"].index(3200)
    scal["ratio_3200"] = scal["dense"]["sec"][-1] / scal["sparse"]["sec"][i3200]
    summary["scaling"] = scal
    viz.fig_scaling(scal, path=str(FIG / "fig8_scaling.png"))
    print(f"       N=102400: {scal['sparse']['sec'][-1]*1e3:.0f} ms, "
          f"{scal['sparse']['mb'][-1]:.2f} MB, E={scal['energy']['102400']:.14f}")

    # ------------------------------------------------------------------ #
    summary["timing"] = {"total_s": time.perf_counter() - t0}
    # ------------------------------------------------------------------ #
    # F. 报告：先跑校验，再渲染 HTML
    # ------------------------------------------------------------------ #
    print("[9/9] 校验并生成报告 ...")
    import subprocess
    import sys

    # 校验失败时停止流水线，避免报告引用上一次的 validation.json。
    chk = subprocess.run(
        [sys.executable, "-X", "utf8", "validate.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if chk.returncode != 0:
        print(chk.stdout[-3000:])
        print(chk.stderr[-3000:], file=sys.stderr)
        raise SystemExit(
            f"validate.py 未通过（返回码 {chk.returncode}）：拒绝用旧的 validation.json 生成报告"
        )
    rows = json.loads((ROOT / "validation.json").read_text(encoding="utf-8"))
    summary["counts"] = {"checks": len(rows), "figures": len(list(FIG.glob("fig[1-8]_*.png")))}
    (ROOT / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    out = report.build(summary, rows, FIG, ROOT / "report.html")
    size = out.stat().st_size / 1024
    print(f"\n完成：{summary['timing']['total_s']:.1f}s，报告 {out.name} ({size:.0f} KB)")
    return summary


if __name__ == "__main__":
    main()
