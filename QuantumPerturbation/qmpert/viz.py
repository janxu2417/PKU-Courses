"""
qmpert.viz
==========

可视化组件。所有函数返回 ``matplotlib.figure.Figure`` 并把图存到 ``path``。
统一样式：浅色、印刷友好、色觉缺陷友好。图中文字用英文/数学符号，
中文图注放在 HTML 报告里（避免 CJK 字体缺失导致的豆腐块）。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator, NullFormatter

__all__ = [
    "use_style",
    "COL",
    "fig_series_overview",
    "fig_convergence",
    "fig_asymptotic",
    "fig_symmetry",
    "fig_sparsity",
    "fig_wavefunction",
    "fig_levels",
    "fig_resummation",
    "fig_scaling",
]

COL = {
    "ink": "#16202b",
    "blue": "#2b5fd9",
    "red": "#c0392b",
    "amber": "#c8811a",
    "teal": "#12766f",
    "purple": "#6b46c1",
    "gray": "#8a94a3",
    "grid": "#e3e7ee",
    "panel": "#fcfcfe",
}
SEQ = [COL["blue"], COL["red"], COL["teal"], COL["amber"], COL["purple"], COL["gray"]]


def use_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": COL["panel"],
            "axes.edgecolor": "#c8cfda",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COL["grid"],
            "grid.linewidth": 0.8,
            "axes.labelcolor": "#2b3644",
            "axes.titlecolor": COL["ink"],
            "axes.titlesize": 13.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 12.6,
            "text.color": COL["ink"],
            "xtick.color": "#5b6675",
            "ytick.color": "#5b6675",
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 10.8,
            "legend.handlelength": 2.2,
            "lines.linewidth": 1.8,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )


def _save(fig, path: str, dpi: int = 165) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _tidy(ax, minor: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if minor:
        if ax.get_xscale() == "linear":
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        if ax.get_yscale() == "linear":
            ax.yaxis.set_minor_locator(AutoMinorLocator())


# --------------------------------------------------------------------------- #
# 0. 单次求解的直接概览：能量系数与归一化态概率
# --------------------------------------------------------------------------- #
def fig_series_overview(series, lam: float, order: int, path: str):
    """保存一张 1×2 图；object 精确系数仅在绘图副本中转为浮点。"""
    from pathlib import Path

    use_style()
    mmax = min(int(order), series.max_order)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.9))

    orders = np.arange(1, mmax + 1)
    values = np.array([float(abs(x)) for x in series.dE[1 : mmax + 1]])
    nonzero = values > 0.0
    if np.any(nonzero):
        ax1.semilogy(orders[nonzero], values[nonzero], "o", color=COL["blue"], ms=4.5)
        if np.any(~nonzero):
            ax1.text(0.03, 0.03, "Zero coefficients omitted", transform=ax1.transAxes, fontsize=9)
    else:
        message = "Order zero only" if mmax == 0 else "All energy corrections vanish"
        ax1.text(0.5, 0.5, message, transform=ax1.transAxes, ha="center", va="center")
    if mmax <= 12:
        ax1.set_xticks(orders)
    else:
        ax1.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
    if mmax:
        ax1.set_xlim(0.5, mmax + 0.5)
    ax1.set_xlabel("order $m$")
    ax1.set_ylabel(r"$|E_n^{(m)}|$")
    ax1.set_title(f"(a) Energy coefficients, $n={series.state}$")
    _tidy(ax1, minor=False)

    # 统一走复数浮点副本，避免 object/SymPy 进入 matplotlib 或 np.linalg.norm。
    vec = np.zeros(series.coeffs.shape[1], dtype=complex)
    for m in range(mmax + 1):
        vec += lam**m * np.asarray(series.coeffs[m], dtype=complex)
    norm = np.linalg.norm(vec)
    probabilities = np.abs(vec / norm) ** 2 if norm else np.abs(vec) ** 2
    top = np.argsort(probabilities)[-12:][::-1]
    ax2.barh(np.arange(top.size), probabilities[top], color=COL["teal"], alpha=0.86)
    ax2.set_yticks(np.arange(top.size), [rf"$|{i}\rangle$" for i in top])
    ax2.invert_yaxis()
    ax2.set_xlabel(r"probability $|c_i|^2$")
    ax2.set_title(rf"(b) Normalized state, $\lambda={lam:g}$, $m={mmax}$")
    _tidy(ax2, minor=False)

    fig.tight_layout()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _save(fig, output)


# --------------------------------------------------------------------------- #
# 1. 收敛性：E(λ) 与误差
# --------------------------------------------------------------------------- #
def fig_convergence(model, series, n: int, lam_max: float, orders: Sequence[int], path: str):
    use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.9, 3.7))

    lam = np.linspace(0.0, lam_max, 90)
    exact = np.array([model.exact_spectrum(l)[n] for l in lam])

    ax1.plot(lam, exact, color=COL["ink"], lw=2.8, label="exact (diagonalisation)", zorder=5)
    for i, o in enumerate(orders):
        ax1.plot(lam, [series.energy(l, o) for l in lam], color=SEQ[i % len(SEQ)],
                 lw=1.5, ls="--", label=f"PT $\\leq {o}$")
    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel(rf"$E_{n}(\lambda)$")
    ax1.set_title(rf"(a) Energy vs coupling  ($n={n}$)")
    ax1.legend(loc="upper left")
    _tidy(ax1)

    ls = np.logspace(np.log10(lam_max) - 2.2, np.log10(lam_max), 70)
    exact_ls = np.array([model.exact_spectrum(l)[n] for l in ls])
    for i, o in enumerate(orders):
        err = [abs(series.energy(l, o) - e) for l, e in zip(ls, exact_ls)]
        ax2.loglog(ls, err, color=SEQ[i % len(SEQ)], lw=1.7, label=f"PT $\\leq {o}$")
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel(r"$|E_{\rm PT}-E_{\rm exact}|$")
    ax2.set_title("(b) Truncation error: power-law scaling")
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.set_ylim(3e-16, 3e-1)
    ax2.legend(loc="lower right")
    _tidy(ax2, minor=False)

    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------------- #
# 2. 渐近级数
# --------------------------------------------------------------------------- #
def fig_asymptotic(model, series: Mapping[int, object], lams: Sequence[float],
                   max_order: int, path: str):
    use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.9, 3.7))

    m = np.arange(1, max_order + 1)
    for i, n in enumerate((0, 1, 2)):
        s = series[n]
        ax1.semilogy(m, np.abs(s.dE[1 : max_order + 1]), "o-", ms=3.0, lw=1.3,
                     color=SEQ[i], label=rf"$|E_{n}^{{(m)}}|$")
    ax1.set_xlabel("order $m$")
    ax1.set_ylabel(r"$|E_n^{(m)}|$")
    ax1.set_title("(a) Coefficients grow like $m!$")
    ax1.text(0.03, 0.94, r"super-exponential:  $|E_n^{(m)}|\sim c^{\,m}\,m!$",
             transform=ax1.transAxes, fontsize=10.4, color=COL["gray"], va="top")
    ax1.legend(loc="lower right", ncol=3)
    _tidy(ax1)

    exact0 = {l: model.exact_spectrum(l)[0] for l in lams}
    for i, l in enumerate(lams):
        errs = np.array([abs(series[0].energy(l, k) - exact0[l]) for k in range(1, max_order + 1)])
        ax2.semilogy(m, errs, "-", lw=1.7, color=SEQ[i], label=rf"$\lambda={l:g}$")
        k = int(np.argmin(errs)) + 1
        ax2.plot(k, errs[k - 1], "o", ms=6, mfc="white", mec=SEQ[i], mew=1.8, zorder=5)
        ax2.annotate(rf"$m^\ast={k}$", (k, errs[k - 1]), textcoords="offset points",
                     xytext=(5, 7), fontsize=10.2, color=SEQ[i])
    ax2.set_xlabel("truncation order $m$")
    ax2.set_ylabel(r"$|E_0^{(0..m)}(\lambda)-E_0^{\rm exact}|$")
    ax2.set_title(r"(b) Optimal truncation $m^\ast\approx 1/(c\lambda)$")
    ax2.legend(loc="lower left")
    _tidy(ax2)

    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------------- #
# 3. 对称性：零阶结构
# --------------------------------------------------------------------------- #
def fig_symmetry(model, series, F_max: float, path: str):
    use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.9, 3.8))

    M = series.max_order
    m = np.arange(1, M + 1)
    vals = np.abs(series.dE[1:])
    zero = vals < 1e-14
    floor = 1e-17

    ax1.set_yscale("log")
    ax1.set_ylim(1e-18, 4.0)
    ax1.vlines(m[~zero], 1e-18, vals[~zero], color=COL["blue"], lw=2.6, zorder=3)
    ax1.plot(m[~zero], vals[~zero], "o", ms=6, color=COL["blue"], zorder=4,
             label=r"$E^{(m)}\neq 0$")
    ax1.plot(m[zero], np.full(int(zero.sum()), floor), "o", ms=6.6, mfc="white",
             mec=COL["red"], mew=1.8, zorder=4, label=r"$E^{(m)}=0$ (selection rule)")
    ax1.axhline(floor, color=COL["red"], lw=0.9, ls=":", alpha=0.65)
    ax1.set_xlabel("order $m$")
    ax1.set_ylabel(r"$|E_0^{(m)}|$")
    ax1.set_title("(a) Odd perturbation: all odd orders vanish")
    ax1.set_xticks(m)
    ax1.annotate("parity-forbidden\n$E^{(m)}\\equiv 0$, $m$ odd", xy=(3, floor), xytext=(3.35, 2e-8),
                 arrowprops=dict(arrowstyle="->", color=COL["red"], lw=1.1), color=COL["red"],
                 fontsize=10.2)
    ax1.annotate(r"$E^{(2)}=-F^2/2$", xy=(2, 0.5), xytext=(2.45, 0.04),
                 arrowprops=dict(arrowstyle="->", color=COL["ink"], lw=1.0), fontsize=10.6)
    ax1.annotate("series terminates\nat $m=2$", xy=(7, floor), xytext=(5.6, 3e-13),
                 color=COL["gray"], fontsize=10.2)
    ax1.legend(loc="upper right")
    _tidy(ax1, minor=False)

    F = np.linspace(-F_max, F_max, 240)
    ax2.plot(F, [series.energy(f, 2) for f in F], color=COL["blue"], lw=6.5, alpha=0.28,
             solid_capstyle="round", label=r"PT $\leq 2$ (coincides with exact)")
    ax2.plot(F, [model.exact_energy(0, f) for f in F], color=COL["ink"], lw=2.6,
             label=r"exact $E_0=1/2-F^2/2$", zorder=4)
    ax2.plot(F, [series.energy(f, 1) for f in F], color=COL["amber"], lw=1.8, ls="--",
             label=r"PT $\leq 1$")
    ax2.set_xlabel("field $F$")
    ax2.set_ylabel(r"$E_0$")
    ax2.set_title("(b) Second order is already exact")
    ax2.legend(loc="lower center")
    _tidy(ax2)

    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------------- #
# 4. 稀疏结构
# --------------------------------------------------------------------------- #
def fig_sparsity(band_model, dense_model, dense_sym, path: str):
    use_style()
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.7), constrained_layout=True)
    raw = dense_model.V
    mask = dense_sym.matrix_mask(raw.shape[0])

    for ax, (V, lab, extra) in zip(
        axes,
        [
            (band_model.V, "(a) $V=x^4$ in the HO basis",
             rf"stored: {100.0*np.count_nonzero(band_model.V)/band_model.V.size:.1f}% of $N^2$"),
            (raw, "(b) dense $V$, raw", r"stored: 100% of $N^2$"),
        ],
    ):
        rel = np.log10(np.abs(V) / np.abs(V).max() + 1e-18)
        im = ax.imshow(rel, cmap="magma", vmin=-16, vmax=0, interpolation="nearest",
                       origin="lower", aspect="auto")
        ax.set_title(lab + "\n" + extra, fontsize=11.6)
        ax.set_xlabel("$n$")
        ax.set_ylabel("$k$")
        ax.grid(False)

    ax = axes[2]
    sub = mask[:56, :56]          # 局部放大，使棋盘格结构可分辨
    ax.imshow(sub.astype(float), cmap=ListedColormap(["#dfe5ef", "#1f2d47"]),
              interpolation="nearest", origin="lower", aspect="auto")
    ax.set_title("(c) parity selection rule\n"
                 rf"allowed: {100.0*mask.sum()/mask.size:.0f}% of $N^2$  (56×56 shown)", fontsize=11.6)
    ax.set_xlabel("$n$")
    ax.set_ylabel("$k$")
    ax.grid(False)

    cb = fig.colorbar(im, ax=list(axes[:2]), fraction=0.022, pad=0.012)
    cb.set_label(r"$\log_{10}(|V_{kn}|/\max|V|)$", fontsize=10.2)
    cb.ax.tick_params(labelsize=9.2)
    _save(fig, path)


# --------------------------------------------------------------------------- #
# 5. 波函数修正
# --------------------------------------------------------------------------- #
def fig_wavefunction(model, series, lam: float, state: int, xg: np.ndarray, orders, path: str):
    from .models import ho_basis_matrix

    use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.9, 3.8), gridspec_kw={"width_ratios": [1.55, 1]})

    psi_exact = model.exact_state(state, lam)
    overlap = np.vdot(series.coeffs[0], psi_exact)
    if overlap != 0:
        psi_exact = psi_exact * (overlap.conjugate() / abs(overlap))
    basis = ho_basis_matrix(xg, model.e0.size)

    ax1.plot(xg, basis.T @ psi_exact, color=COL["ink"], lw=3.0, label="exact", zorder=5)
    for i, o in enumerate(orders):
        v = series.normalized_state(lam, o)
        ax1.plot(xg, basis.T @ v, lw=1.4, ls=["--", "-.", ":"][i % 3], color=SEQ[(i + 1) % len(SEQ)],
                 label=f"PT $\\leq {o}$")
    ax1.set_xlim(-4.2, 4.2)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel(r"$\psi_0(x)$")
    ax1.set_title(rf"(a) Wavefunction at $\lambda={lam:g}$")
    ax1.legend(loc="upper right")
    _tidy(ax1)

    mmax = min(8, series.max_order)
    l2 = [np.linalg.norm(series.normalized_state(lam, o) - psi_exact) for o in range(0, mmax + 1)]
    displayed = np.maximum(l2, 1e-17)
    ks = np.arange(0, mmax + 1)
    ax2.semilogy(ks, displayed, "o-", color=COL["red"], ms=4.6, lw=1.6)
    kbest = int(np.argmin(l2))
    ax2.plot(ks[kbest], displayed[kbest], "o", ms=9, mfc="white", mec=COL["red"], mew=1.6, zorder=5)
    ax2.annotate(rf"$m^\ast={kbest}$", (ks[kbest], displayed[kbest]), textcoords="offset points",
                 xytext=(8, -14), fontsize=8.8, color=COL["red"])
    ax2.set_xlabel("truncation order $m$")
    ax2.set_ylabel(r"$\Vert \psi_{\rm PT}-\psi_{\rm exact}\Vert_2$")
    ax2.set_title("(b) State-vector error vs order")
    _tidy(ax2, minor=False)

    fig.tight_layout()
    _save(fig, path)
    return l2


# --------------------------------------------------------------------------- #
# 6. 能级图
# --------------------------------------------------------------------------- #
def fig_levels(model, series_list, lam_max: float, orders=(1, 2), path: str = ""):
    use_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    lam = np.linspace(0, lam_max, 80)
    spec = np.array([model.exact_spectrum(l) for l in lam])
    ns = [s.state for s in series_list]
    for i, s in enumerate(series_list):
        c = SEQ[i % len(SEQ)]
        e0 = s.unperturbed_energy
        ax.plot(lam, spec[:, s.state] - e0, color=c, lw=2.5, zorder=4)
        ax.plot(lam, [s.energy(l, orders[0]) - e0 for l in lam], color=c, lw=1.3, ls="--", alpha=0.95)
        ax.plot(lam, [s.energy(l, orders[-1]) - e0 for l in lam], color=c, lw=1.2, ls=":", alpha=0.95)
        ax.annotate(f"$n={s.state}$", (lam_max, spec[-1, s.state] - e0), textcoords="offset points",
                    xytext=(8, 0), fontsize=9, color=c, va="center", annotation_clip=False)

    handles = [
        Line2D([], [], color=COL["ink"], lw=2.5, label="exact"),
        Line2D([], [], color=COL["ink"], lw=1.3, ls="--", label=rf"PT $\leq {orders[0]}$"),
        Line2D([], [], color=COL["ink"], lw=1.2, ls=":", label=rf"PT $\leq {orders[-1]}$"),
    ]
    ax.legend(handles=handles, loc="upper left")
    ax.set_xlim(0, lam_max)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$E_n(\lambda)-E_n^{(0)}$")
    ax.set_title("Level shifts vs coupling")
    _tidy(ax)
    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------------- #
# 7. 重求和：发散级数 vs Padé / Borel–Padé
# --------------------------------------------------------------------------- #
def fig_resummation(model, series, coeffs, path: str, state: int = 0,
                    lam_scan=(0.01, 1.0), lam_order: float = 0.05):
    from .resum import BorelPade, PadeTable

    use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.85))

    tab = PadeTable(coeffs, max_order=20)
    borel = BorelPade(coeffs, 16)

    # ---------------- (a) 误差随 λ ----------------
    lams = np.logspace(np.log10(lam_scan[0]), np.log10(lam_scan[1]), 44)
    exact = np.array([model.exact_spectrum(l)[state] for l in lams])
    m_max = series.max_order

    e_opt = np.array([
        np.min([abs(series.energy(l, k) - ex) for k in range(1, m_max + 1)])
        for l, ex in zip(lams, exact)
    ])
    e_10 = np.array([abs(series.energy(l, 10) - ex) for l, ex in zip(lams, exact)])
    e_pade = np.array([abs(tab.best(l)[0] - ex) for l, ex in zip(lams, exact)])
    e_borel = []
    for l, ex in zip(lams, exact):
        try:
            e_borel.append(abs(borel(l) - ex))
        except ValueError:
            e_borel.append(np.nan)
    e_borel = np.array(e_borel)

    ax1.loglog(lams, np.maximum(e_opt, 2e-16), color=COL["gray"], lw=2.0,
               label=r"truncation, best $m^\ast$")
    ax1.loglog(lams, np.maximum(e_10, 2e-16), color=COL["amber"], lw=1.5, ls="--",
               label=r"truncation, $m=10$")
    ax1.loglog(lams, np.maximum(e_pade, 2e-16), color=COL["blue"], lw=2.2,
               label=r"Padé $[k/k]$")
    ax1.loglog(lams, np.maximum(e_borel, 2e-16), color=COL["teal"], lw=2.4,
               label=r"Borel–Padé $[16/16]$")

    ax1.axvspan(lam_scan[0], 0.02, color=COL["blue"], alpha=0.05, lw=0)
    ax1.text(0.0112, 2e-1, "perturbative\nwindow", color="#8b96a6", fontsize=9.0,
             rotation=90, va="top", ha="left")
    ax1.axhline(1e-12, color=COL["gray"], lw=0.9, ls=":")
    ax1.text(0.93, 1.9e-12, r"$10^{-12}$", color=COL["gray"], fontsize=9.6, va="bottom", ha="right")
    ax1.set_xlim(lam_scan[0], lam_scan[1])
    ax1.set_ylim(1.2e-16, 3.0)
    ax1.set_xlabel(r"coupling $\lambda$")
    ax1.set_ylabel(r"$|E_0^{\rm approx}-E_0^{\rm exact}|$")
    ax1.set_title("(a) Accuracy across coupling")
    ax1.legend(loc="upper center", fontsize=10.0, handlelength=2.0, ncol=2,
               bbox_to_anchor=(0.5, -0.20), columnspacing=1.4, labelspacing=0.35)
    _tidy(ax1, minor=False)

    # ---------------- (b) 误差随阶数 ----------------
    ex = model.exact_spectrum(lam_order)[state]
    ms = np.arange(1, m_max + 1)
    e_tr = np.array([abs(series.energy(lam_order, k) - ex) for k in ms])
    ks, e_pd = tab.errors(lam_order, ex)
    kb = [k for k in range(4, 17)]
    e_bp = []
    for k in kb:
        try:
            e_bp.append(abs(BorelPade(coeffs, k)(lam_order) - ex))
        except ValueError:
            e_bp.append(np.nan)
    e_bp = np.array(e_bp)

    ax2.semilogy(ms, np.maximum(e_tr, 2e-16), color=COL["gray"], lw=1.8,
                 marker="o", ms=3.0, mfc="white", mew=0.8, label="truncation")
    ax2.semilogy(ks, np.maximum(e_pd, 2e-16), color=COL["blue"], lw=2.2,
                 marker="s", ms=3.4, label=r"Padé $[k/k]$")
    ax2.semilogy(kb, np.maximum(e_bp, 2e-16), color=COL["teal"], lw=2.2,
                 marker="^", ms=3.6, label=r"Borel–Padé $[k/k]$")
    ax2.axvline(16.4, color=COL["teal"], lw=1.0, ls=":")
    ax2.text(16.95, 3e12, "spurious\npoles $\\rightarrow$", color=COL["teal"],
             fontsize=9.2, va="top")

    ax2.set_xlim(0, 30)
    ax2.set_ylim(1.2e-16, 1e14)
    ax2.set_xlabel("order")
    ax2.set_ylabel("error")
    ax2.set_title(rf"(b) Order-by-order at $\lambda={lam_order:g}$")
    ax2.legend(loc="upper left", fontsize=10.2, bbox_to_anchor=(0.02, 0.99))
    _tidy(ax2, minor=False)

    fig.tight_layout()
    _save(fig, path)


# --------------------------------------------------------------------------- #
# 8. 规模：稠密 vs 稀疏
# --------------------------------------------------------------------------- #
def fig_scaling(scaling: dict, path: str):
    """存储与耗时随基组维度的变化（双对数）。

    存储量是**解析量**（8N² 字节 / 12·nnz 字节），故稠密曲线可外推到测量
    范围之外；耗时才需要实测。
    """
    use_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # ---------------- (a) 存储 ----------------
    Nd = np.array(scaling["dense"]["N"], dtype=float)
    Ns = np.array(scaling["sparse"]["N"], dtype=float)
    dense_mb = np.array(scaling["dense"]["mb"], dtype=float)
    sparse_mb = np.array(scaling["sparse"]["mb"], dtype=float)

    Nfull = np.logspace(np.log10(80), np.log10(1.02e5), 80)
    dense_full = 8.0 * Nfull**2 / 1e6
    left = Nfull <= Nd.max()
    ax1.loglog(Nfull[left], dense_full[left], color=COL["amber"], lw=2.4, label="dense $8N^2$ B")
    ax1.loglog(Nfull[~left], dense_full[~left], color=COL["amber"], lw=1.5,
               ls=(0, (4.5, 2.2)), alpha=0.8)
    ax1.loglog(Ns, sparse_mb, color=COL["blue"], lw=2.4, label="sparse $12\\cdot$nnz B")

    ax1.axhline(32e3, color=COL["gray"], lw=1.0, ls=":")
    ax1.text(86, 4.0e4, "32 GB", color=COL["gray"], fontsize=9.8, va="bottom")
    ax1.axvspan(6.3e4, 1.3e5, color=COL["amber"], alpha=0.07, lw=0)

    i_last = int(np.argmin(np.abs(Ns - 1.02e5)))
    N_last = float(Ns[i_last])
    ax1.plot([N_last], [8 * N_last**2 / 1e6], "o", ms=4.4, color=COL["amber"],
             mfc="white", mew=1.2)
    ax1.text(0.035, 0.70,
             f"at $N\\approx10^5$\ndense  {8*N_last**2/1e9:.0f} GB"
             f"\nsparse  {sparse_mb[i_last]:.1f} MB",
             transform=ax1.transAxes, fontsize=9.8, va="top", ha="left",
             linespacing=1.55, color=COL["ink"])

    ax1.set_xlim(80, 1.3e5)
    ax1.set_ylim(1e-3, 3e5)
    ax1.set_xlabel("basis size $N$")
    ax1.set_ylabel("storage (MB)")
    ax1.set_title("(a) Memory footprint")
    ax1.legend(loc="lower right", fontsize=10.0)
    _tidy(ax1, minor=False)

    # ---------------- (b) 耗时 ----------------
    td = np.array(scaling["dense"]["sec"], dtype=float)
    ts = np.array(scaling["sparse"]["sec"], dtype=float)
    ax2.loglog(Nd, td, color=COL["amber"], lw=2.2, marker="o", ms=4.0,
               mfc="white", mew=1.1, label="dense")
    ax2.loglog(Ns, ts, color=COL["blue"], lw=2.2, marker="s", ms=4.0,
               mfc="white", mew=1.1, label="sparse")

    x0, y0 = Ns[2], ts[2]
    ax2.loglog(Ns, y0 * (Ns / x0), color=COL["gray"], lw=1.0, ls="--")
    ax2.text(3.2e4, y0 * 3.2e4 / x0 * 2.4, r"$\propto N$", color=COL["gray"], fontsize=9.8)
    x0d, y0d = Nd[2], td[2]
    ax2.loglog(Nd, y0d * (Nd / x0d) ** 2, color=COL["gray"], lw=1.0, ls=":")
    ax2.text(500, y0d * (500 / x0d) ** 2 * 6.0, r"$\propto N^2$", color=COL["gray"], fontsize=9.8)

    ax2.set_xlim(80, 1.3e5)
    ax2.set_ylim(3e-5, 60)
    ax2.set_xlabel("basis size $N$")
    ax2.set_ylabel("time for 40 orders (s)")
    ax2.set_title("(b) Wall time")
    ax2.legend(loc="upper left", fontsize=10.0)
    _tidy(ax2, minor=False)

    fig.tight_layout()
    _save(fig, path)
