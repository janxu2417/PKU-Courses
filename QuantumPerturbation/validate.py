"""
validate.py —— 正确性校验：解析解 / 高精度数值对角化 / 对称性自洽性

运行：  python3 validate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qmpert import (
    DegeneratePerturbationTheory,
    DegenerateSpectrumError,
    DeltaSymmetry,
    ParitySymmetry,
    PerturbationTheory,
    ResidualDegeneracyError,
    anharmonic_oscillator,
    displaced_oscillator,
    random_model,
    two_level,
)

OK, FAIL = "PASS", "FAIL"
rows: list[tuple[str, str, str, str]] = []


def record(name: str, err: float, tol: float, detail: str = "") -> None:
    rows.append((name, f"{err:.2e}", f"{tol:.0e}", OK if err <= tol else FAIL))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# --------------------------------------------------------------------------- #
# 1. 四次非谐振子：与文献解析系数对照
#    E_n⁽¹⁾ = (3/4)(2n²+2n+1)
#    E_n⁽²⁾ = -(1/8)(34n³+51n²+59n+21)
# --------------------------------------------------------------------------- #
section("1. 四次非谐振子 x^4：解析系数")
m = anharmonic_oscillator(N=220, power=4)
pt = PerturbationTheory(m.e0, m.V, name=m.name)
print(f"{'n':>3} {'E^(1) 数值':>14} {'E^(1) 解析':>14} {'E^(2) 数值':>14} {'E^(2) 解析':>14}")
for n in range(4):
    s = pt.solve(n, 6)
    a1 = 0.75 * (2 * n**2 + 2 * n + 1)
    a2 = -(34 * n**3 + 51 * n**2 + 59 * n + 21) / 8.0
    print(f"{n:3d} {s.dE[1]:14.10f} {a1:14.10f} {s.dE[2]:14.10f} {a2:14.10f}")
    record(f"x^4  E{n}^(1) vs 解析", abs(s.dE[1] - a1), 1e-10)
    record(f"x^4  E{n}^(2) vs 解析", abs(s.dE[2] - a2), 1e-8)

# 高精度数值基准（同一截断空间内严格对角化）
# λ 很小时级数在截断前已收敛到机器精度（渐近级数的最佳截断阶 m* ≈ 1/(cλ)）
lam = 1e-3
s_small = pt.solve(0, 12)
e_small = m.exact_spectrum(lam)[0]
errs_small = [abs(s_small.energy(lam, k) - e_small) for k in (2, 4, 6, 8, 10, 12)]
print("λ=1e-3 各阶截断误差:", " ".join(f"{v:.1e}" for v in errs_small))
record("x^4 λ=1e-3 误差随阶数下降", 0.0 if all(b <= a for a, b in zip(errs_small, errs_small[1:])) else 1.0, 0.0)
record("x^4 λ=1e-3 12 阶达机器精度", errs_small[-1], 1e-13)

# 渐近性：λ 较大时误差先减后增，存在"最佳截断阶"
s = pt.solve(0, 40)
lam_big = 0.05
e_big = m.exact_spectrum(lam_big)[0]
errs = np.array([abs(s.energy(lam_big, k) - e_big) for k in range(1, 41)])
kbest = int(np.argmin(errs)) + 1
print(f"λ=0.05: 最佳截断阶 m*={kbest}, 误差 {errs[kbest-1]:.2e}; 1 阶误差 {errs[0]:.2e}; 40 阶误差 {errs[-1]:.2e}")
record("渐近性：最佳截断优于低阶", 0.0 if errs[kbest-1] < errs[0] else 1.0, 0.0)
record("渐近性：高阶最终发散", 0.0 if errs[-1] > errs[kbest-1] * 1e3 else 1.0, 0.0)

# 级数增长：|E^(m)| 应随 m 阶乘增长
mag = np.abs(s.dE[1:11])
ratio_growth = mag[1:] / mag[:-1]
print("|E^(m)| =", np.array2string(mag, precision=3, suppress_small=False))
print("相邻比 =", np.array2string(ratio_growth, precision=2))
record("x^4  |E^(m)| 单调增长(渐近性)", 0.0 if np.all(np.diff(mag) > 0) else 1.0, 0.0)

# --------------------------------------------------------------------------- #
# 2. 位移谐振子 −Fx：宇称 ⇒ 奇阶为零；二阶精确终止
# --------------------------------------------------------------------------- #
section("2. 位移谐振子 H0 − F x：奇阶为零 + 二阶精确")
md = displaced_oscillator(N=200)
sym = ParitySymmetry(state_parities=(-1) ** np.arange(md.e0.size), perturbation_parity=-1)
ptd = PerturbationTheory(md.e0, md.V, symmetry=sym, name=md.name)
print("对称性自洽性（禁戒元最大绝对值）：", f"{sym.validate(md.V):.2e}")
sd = ptd.solve(0, 10)
print("E^(m), m=1..8 :", np.array2string(sd.dE[1:9], precision=12))
import math

record("Stark  E^(1)=0", abs(sd.dE[1]), 1e-14)
record("Stark  E^(2)=-1/2", abs(sd.dE[2] + 0.5), 1e-14)
record("Stark  奇数阶全为 0", max(abs(sd.dE[m]) for m in (1, 3, 5, 7, 9)), 1e-14)
record("Stark  m≥3 全为 0", max(abs(sd.dE[m]) for m in range(3, 11)), 1e-13)
pred = {1, 3, 5, 7, 9}
meas = set(sd.measured_zero_orders())
print("对称性预测零阶:", sorted(pred), " 实测零阶:", sorted(meas), " 额外为零:", sorted(meas - pred))
record("Stark  预测零阶 ⊆ 实测零阶", 0.0 if pred <= meas else 1.0, 0.0)
for F in (0.05, 0.2, 0.5, 1.0):
    err = abs(sd.energy(F, order=2) - md.exact_energy(0, F))
    record(f"Stark  F={F:<4} 二阶 vs 解析", err, 1e-12)

# --------------------------------------------------------------------------- #
# 3. 二能级：任意阶 vs 闭式 √(1+x) 展开
# --------------------------------------------------------------------------- #
section("3. 二能级系统：任意阶 vs 闭式")
delta, v = 1.0, 0.02
mt = two_level(delta)
ptt = PerturbationTheory(mt.e0, mt.V)
st = ptt.solve(0, 14)
series_coeff = []
for k in range(1, 15):
    # Δ/2 − Δ/2·√(1+4v²/Δ²) 中 v^(2k) 的系数
    from math import prod

    gc = prod(0.5 - j for j in range(k)) / prod(range(1, k + 1))   # 广义二项式 C(1/2,k)
    c = -delta / 2 * gc * (4 / delta**2) ** k
    series_coeff.append(c)
series_coeff = np.array(series_coeff)
print("E^(2k) 数值:", np.array2string(st.dE[2:12:2], precision=10))
print("E^(2k) 解析:", np.array2string(series_coeff[:5], precision=10))
record("二能级 偶数阶系数 vs 闭式", np.max(np.abs(st.dE[2:12:2] - series_coeff[:5])), 1e-12)
record("二能级 奇数阶 = 0", np.max(np.abs(st.dE[1:12:2])), 1e-14)
record("二能级 E(v=0.02) 14阶 vs 解析", abs(st.energy(0.02, 14) - mt.exact_energy(0, 0.02)), 1e-15)

# --------------------------------------------------------------------------- #
# 4. 随机非简并模型：与数值对角化对照（小 λ，高阶）
# --------------------------------------------------------------------------- #
section("4. 随机模型：1..8 阶 vs 数值精确解")
worst = 0.0
for seed in range(5):
    mr = random_model(N=10, seed=seed)
    ptr = PerturbationTheory(mr.e0, mr.V)
    lam_r = 1e-3
    ex = mr.exact_spectrum(lam_r)
    for n in range(10):
        sr = ptr.solve(n, 8)
        err = abs(sr.energy(lam_r, order=8) - ex[n])
        worst = max(worst, err)
print("最大偏差：", f"{worst:.2e}")
record("随机模型 8 阶 vs 精确", worst, 1e-12)

# --------------------------------------------------------------------------- #
# 5. 复数 Hermitian 微扰（dtype 支持）
# --------------------------------------------------------------------------- #
section("5. 复数 Hermitian 微扰")
rng = np.random.default_rng(7)
A = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
Vc = (A + A.conj().T) / 2
e0c = np.array([0.0, 1.3, 2.9, 4.1, 6.0, 7.7])
ptc = PerturbationTheory(e0c, Vc)
sc = ptc.solve(2, 6)
lams = 1e-4
exact_c = np.linalg.eigvalsh(np.diag(e0c) + lams * Vc)[2]
record("复数 V 6 阶 vs 精确", abs(sc.energy(lams, 6) - exact_c), 1e-14)

# --------------------------------------------------------------------------- #
# 6. 简并守卫 + 选择定则掩码
# --------------------------------------------------------------------------- #
section("6. 简并守卫与选择定则")
try:
    PerturbationTheory(np.array([0.0, 1.0, 1.0]), np.zeros((3, 3))).solve(1, 2)
    record("简并检测", 1.0, 0.0)
except DegenerateSpectrumError as exc:
    print("捕获:", exc)
    record("简并检测（应抛异常）", 0.0, 0.0)

sym4 = DeltaSymmetry({0, 2, -2, 4, -4}, name="Δn ∈ {0,±2,±4}")
print("x^4 对 Δ 选择定则的违反量：", f"{sym4.validate(m.V):.2e}")
record("x^4 选择定则自洽", sym4.validate(m.V), 1e-14)
pt4 = PerturbationTheory(m.e0, m.V, symmetry=sym4)
d4 = pt4.diagnostics()
print("非零元：", d4["nnz_raw"], "→", d4["nnz_used"], f"（保留 {100*d4['nnz_ratio']:.1f}%）")
record("掩码不改变物理结果", abs(pt4.solve(0, 6).energy(0.02, 6) - s.energy(0.02, 6)), 1e-14)

# --------------------------------------------------------------------------- #
section("7. 稠密微扰上的对称性掩码")
from qmpert import gaussian_bump

mg = gaussian_bump(N=100)
symg = ParitySymmetry((-1) ** np.arange(mg.e0.size), +1)
ptg = PerturbationTheory(mg.e0, mg.V, symmetry=symg)
dg = ptg.diagnostics()
print(f"禁戒元最大绝对值 {symg.validate(mg.V):.1e}；非零元 {dg['nnz_raw']} → {dg['nnz_used']}"
      f"（保留 {100*dg['nnz_ratio']:.0f}%）")
record("稠密 V 的宇称违反量", symg.validate(mg.V), 1e-14)
record("掩码剔除比例 = 50%", abs(dg["nnz_ratio"] - 0.5), 1e-12)
ptg0 = PerturbationTheory(mg.e0, mg.V, symmetry=None)
record("掩码前后能量一致", abs(ptg.solve(0, 6).energy(0.1, 6) - ptg0.solve(0, 6).energy(0.1, 6)), 1e-14)

# --------------------------------------------------------------------------- #
section("8. 重求和：Padé 与 Borel–Padé")
from qmpert.resum import BorelPade, PadeTable

s40 = pt.solve(0, 40)
c40 = np.concatenate(([s40.unperturbed_energy], np.asarray(s40.dE[1:], float)))
tab = PadeTable(c40, max_order=20)

# (a) 代数恒等式：Q·Σc − P 在高精度下为零（舍入到 double 后无法检验）
print(f"Padé [20/20] 匹配残差（高精度）：{tab.approx[20].residual_mp:.1e}")
record("Padé [20/20] 匹配残差", tab.approx[20].residual_mp, 1e-80)

# (b) 小 λ：Padé 必须退化为 Taylor 部分和
record("Padé 小 λ 退化为 Taylor", abs(tab.at(0.01, 20) - s40.energy(0.01, 20)), 1e-12)

# (c) 截断已失效的区间
for lam, tol in ((0.1, 1e-9), (0.2, 1e-7), (0.5, 1e-5)):
    ex = m.exact_spectrum(lam)[0]
    record(f"Padé λ={lam} 超越最优截断", abs(tab.best(lam)[0] - ex), tol)

# (d) Borel–Padé
bp16 = BorelPade(c40, 16)
for lam, tol in ((0.05, 1e-12), (0.2, 1e-9), (0.5, 1e-6)):
    ex = m.exact_spectrum(lam)[0]
    record(f"Borel–Padé λ={lam}", abs(bp16(lam) - ex), tol)

# (e) 极点筛查：更高阶的 Borel–Padé 有伪极点，必须被拒绝
bad = BorelPade(c40, 20)
print(f"Borel [20/20] 在 λ=0.2 的危险极点：{bad.dangerous_poles(0.2)}")
record("Borel 伪极点检出（k=20）", 0.0 if not bad.is_safe(0.2) else 1.0, 0.0)

# (f) λ→0 极限应回到未微扰能量
record("Borel–Padé λ→0 趋于 E0", abs(bp16(1e-6) - m.exact_spectrum(1e-6)[0]), 1e-10)

# --------------------------------------------------------------------------- #
section("9. 精确算术：与 Bender–Wu 系数逐位对照")
import sympy as sp
from fractions import Fraction

from qmpert.models import symbolic_anharmonic

e0s, Vs = symbolic_anharmonic(N=20)
pt_sym = PerturbationTheory(np.array(e0s, dtype=object), np.array(Vs.tolist(), dtype=object))
s_sym = pt_sym.solve(0, 6)

# 文献值：E0 = 1/2 + 3/4 λ − 21/8 λ² + 333/16 λ³ − 30885/128 λ⁴
#               + 916731/256 λ⁵ − 65518401/1024 λ⁶ + …
known = [Fraction(3, 4), Fraction(-21, 8), Fraction(333, 16),
         Fraction(-30885, 128), Fraction(916731, 256), Fraction(-65518401, 1024)]
for m, ref in enumerate(known, start=1):
    got = Fraction(sp.nsimplify(s_sym.dE[m]))
    record(f"E0^({m}) 精确值 = {ref}", 0.0 if got == ref else 1.0, 0.0)
print("精确系数：", [str(sp.nsimplify(x)) for x in s_sym.dE[1:7]])

# 与浮点后端的一致（同一物理，两条独立路径）
s_float = pt.solve(0, 6)
rel = max(abs(float(a) - b) / abs(b) for a, b in zip(s_sym.dE[1:7], s_float.dE[1:7]))
record("精确 vs 浮点系数（相对）", rel, 1e-12)

# 精确求值：λ=1/10 处的截断能量应是有理数
val = s_sym.energy_exact(Fraction(1, 10))
print(f"E0(λ=1/10) 精确 = {val} = {float(val):.15f}")
record("精确能量返回到 Fraction", 0.0 if isinstance(val, Fraction) else 1.0, 0.0)
record("精确 vs 浮点能量 λ=1/10", abs(float(val) - s_float.energy(0.1, 6)), 1e-13)

# --------------------------------------------------------------------------- #
section("10. 稀疏后端：与稠密逐元对照")
from qmpert.models import banded_anharmonic

# (a) 两条独立构造路径给出的 V 必须相同
for N in (60, 200):
    ms = banded_anharmonic(N=N)
    md = anharmonic_oscillator(N=N)
    dv = np.abs(ms.V.toarray() - md.V).max() / np.abs(md.V).max()
    record(f"稀疏 x^4 vs 稠密 x^4 (N={N})", dv, 1e-14)

# (b) 级数系数一致
ms, md = banded_anharmonic(N=200), anharmonic_oscillator(N=200)
ss = PerturbationTheory(ms.e0, ms.V).solve(0, 24)
sd = PerturbationTheory(md.e0, md.V).solve(0, 24)
rel = np.max(np.abs(ss.dE - sd.dE)) / np.max(np.abs(sd.dE))
record("稀疏 vs 稠密 24 阶系数（相对）", rel, 1e-12)
print(f"稀疏 N=200：nnz={ms.V.nnz}，稠密需 {200*200}；"
      f"存储 {PerturbationTheory(ms.e0, ms.V).diagnostics()['stored_mb']:.3f} MB")

# (c) 稀疏路径的选择定则：结构掩码与稠密掩码等价
sym = ParitySymmetry((-1) ** np.arange(600), +1)
mbig = banded_anharmonic(N=600)
engine = PerturbationTheory(mbig.e0, mbig.V, symmetry=sym)
mask_sparse = engine.v.astype(bool).toarray()
mask_dense = sym.matrix_mask(600) & (mbig.V.toarray() != 0)
record("稀疏结构掩码 ≡ 稠密掩码", 0.0 if np.array_equal(mask_sparse, mask_dense) else 1.0, 0.0)
record("稀疏掩码后禁戒元为零", float(np.abs(engine.v.toarray()[~sym.matrix_mask(600)]).max()), 0.0)

# (d) 大基组：精确谱与低维参考一致（移位反演，不建稠密矩阵）
ref = anharmonic_oscillator(N=200).exact_spectrum(0.02)[0]
for N in (3200, 25600):
    e_big = banded_anharmonic(N=N).exact_spectrum(0.02, k=1)[0]
    record(f"N={N} 精确谱 vs N=200 参考", abs(e_big - ref), 1e-12)
print(f"N=200 参考 E0 = {ref:.15f}")

# (e) 40 阶层级在大基组上仍给出同一能量（基组已收敛）
s_big = PerturbationTheory(banded_anharmonic(N=25600).e0,
                           banded_anharmonic(N=25600).V).solve(0, 40)
record("N=25600 的 E(λ=0.02, 20 阶) vs N=200",
       abs(s_big.energy(0.02, 20) - PerturbationTheory(md.e0, md.V).solve(0, 20).energy(0.02, 20)),
       1e-13)

# --------------------------------------------------------------------------- #
section("11. 二阶简并微扰：非简并极限、二阶选基与残余简并")
e0 = np.array([0.0, 2.0, 5.0])
rng = np.random.default_rng(2026)
matrix = rng.normal(size=(3, 3))
V = (matrix + matrix.T) / 2
branches = DegeneratePerturbationTheory(e0, V).solve().branches
errors = []
for n, branch in enumerate(branches):
    outside = np.arange(len(e0)) != n
    c1_ref = np.zeros_like(e0)
    c1_ref[outside] = V[outside, n] / (e0[n] - e0[outside])
    errors.extend((abs(branch.dE1 - V[n, n]), abs(branch.dE2 - V[n] @ c1_ref),
                   np.max(np.abs(branch.c1 - c1_ref))))
record("简并求解器：非简并极限", max(errors), 1e-13)

rt2 = np.sqrt(2)
V_lecture = [[0, rt2, 2 * rt2], [rt2, 0, 2], [2 * rt2, 2, 0]]
branches = DegeneratePerturbationTheory([0, 1, 1], V_lecture).solve().branches
record("简并三能级：W₂ 给出 (-10, 1, 9)",
       np.max(np.abs(np.array([b.dE2 for b in branches]) - [-10, 1, 9])), 1e-13)

V_coupled = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]
branch = DegeneratePerturbationTheory([0, 0, 2, 3], V_coupled).solve().branches[0]
error = max(abs(branch.dE2 + 0.5), np.max(np.abs(branch.c1 - [0, -1, -0.5, 0])))
record("二阶劈裂：K₃ 确定一阶态矢", error, 1e-13)

try:
    DegeneratePerturbationTheory([0, 0, 2, 2], V_coupled).solve()
except ResidualDegeneracyError:
    record("二阶仍简并：抛出 ResidualDegeneracyError", 0.0, 0.0)
else:
    record("二阶仍简并：抛出 ResidualDegeneracyError", 1.0, 0.0)

# --------------------------------------------------------------------------- #
print("\n" + "=" * 78)
print(f"{'检验项':<38}{'误差':>12}{'容差':>10}{'结论':>8}")
print("-" * 78)
for name, err, tol, verdict in rows:
    print(f"{name:<38}{err:>12}{tol:>10}{verdict:>8}")
n_fail = sum(1 for r in rows if r[3] == FAIL)
print("-" * 78)
print(f"合计 {len(rows)} 项，失败 {n_fail} 项。")

Path(__file__).with_name("validation.json").write_text(
    json.dumps(
        [{"name": n, "err": e, "tol": t, "verdict": v} for n, e, t, v in rows],
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
if n_fail:
    raise SystemExit(1)
