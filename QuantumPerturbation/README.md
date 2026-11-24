# qmpert

Rayleigh–Schrödinger 微扰论（RSPT）求解工具箱：任意阶**非简并**递推、**二阶简并**微扰（能量到二阶、态矢到一阶）、逐阶范数归一化、精确符号输出与直接出图；另带选择定则、稀疏后端和 Padé / Borel–Padé 重求和。

| 项 | 内容 |
| --- | --- |
| 版本 | 1.2.0 |
| 依赖 | Python ≥ 3.9。安装项目时按 `pyproject.toml` 安装 NumPy、SciPy、Matplotlib、mpmath 和 SymPy |
| 课程作业 | 本地 `作业报告.md` 汇总算法、讲义三个例题和复现命令（过程文档，未随仓库发布） |
| 扩展演示 | `python run_all.py` 生成自包含 `report.html`，含四次非谐振子等额外模型 |

## 特性

- **任意阶** ：每阶做一次 $V\mathbf c^{(m-1)}$ 与低阶系数反馈求和。阶数没有结构上的上限。
- **三套后端** ：`float` / `complex` / 精确有理数（`Fraction` 与 `sympy.Rational` 的对象数组），后者可把 $E_n^{(m)}$ 算成精确分数。
- **两种归一化** ：保留中间归一化递推，同时用 `normalized_coeffs()` 给出与老师式 (24) 一致的逐阶范数归一化态矢系数。
- **二阶简并微扰** ：先对角化 $PVP$。一阶仍简并时，再在剩余子空间内对角化二阶有效矩阵。支持数值与 SymPy 精确矩阵。
- **直接出图** ：例题和 CLI 可保存能量、系数或态矢图，不必先生成完整 HTML 报告。
- **选择定则** ：宇称、$\Delta n$、一般量子数、复合规则。既能校验 $V$ 是否自洽，也能**先验**判定哪些阶必然为零，还能把矩阵掩码成严格分块。
- **稀疏** ：`V` 为 `scipy.sparse` 时矩阵–向量乘降到 $O(\mathrm{nnz})$（另有与 $N$ 成正比的反馈项，见 [`docs/algorithm.md`](docs/algorithm.md) §4.4），$N=10^5$ 量级仍可跑 40 阶。
- **重求和** ：Padé（mpmath 高精度构造）与 Borel–Padé（自带伪极点筛查），把可用区间从 $\lambda\lesssim0.02$ 推到 $\lambda\sim1$。
- **可复现**：`run_all.py` 重算全部图表、数值与报告，`validate.py` 单独运行正确性校验。

## 安装

```bash
pip install -e .          # 运行时依赖
pip install -e ".[dev]"   # 额外安装 pytest
```

源码中的 NumPy 稠密数值递推可以独立运行。SciPy 用于稀疏后端、势阱例题的求根和重求和，Matplotlib 用于绘图，SymPy 用于符号矩阵和精确归一化，mpmath 用于高精度重求和。

## 快速开始

```python
import numpy as np
from qmpert import PerturbationTheory, anharmonic_oscillator

m  = anharmonic_oscillator(N=200)                  # H = p²/2 + x²/2 + λx⁴，ℏ=m=ω=1
pt = PerturbationTheory(m.e0, m.V)                 # H₀ 的本征值 + V 的矩阵表示
s  = pt.solve(state=0, max_order=10)

s.dE[1], s.dE[2]              # (0.75, -2.625) = 3/4, -21/8
s.energy(lam=0.05, order=4)   # 截断到 4 阶的能量
s.normalized_coeffs()         # 范数归一化的逐阶态矢系数，用于对照讲义
s.normalized_state(0.05, 3)   # 给定 λ 后，把 3 阶截断态整体归一化
```

加一条选择定则，可直接得到“哪些阶必然为零”：

```python
from qmpert import displaced_oscillator, ParitySymmetry

md  = displaced_oscillator(N=120)                  # H = H₀ − Fx，奇宇称微扰
sym = ParitySymmetry((-1) ** np.arange(md.e0.size), perturbation_parity=-1)
sd  = PerturbationTheory(md.e0, md.V, symmetry=sym).solve(0, 10)

sorted(sd.zeros_predicted)    # [1, 3, 5, 7, 9]：奇阶恒为零
sd.dE[2]                      # -0.5，且 m≥3 全为零（级数二阶精确终止）
```

### 把讲义例题搬进来

`examples/` 下有三个完整示范：两个对应《第 12 章 · §4 非简并微扰的两个例题》，另一个对应 §7.1 的三能级简并例题。每个脚本都直接打印与讲义公式的对照：

| 脚本 | 模型 | 对照的式子 |
| --- | --- | --- |
| `examples/ex1_two_level.py` | $H=(\delta/2)\sigma_z+(\Omega/2)\sigma_x$ | (44)–(47)：精确分数、态混合概率和截断阶数误差 |
| `examples/ex2_delta_well.py` | 无限深方势阱 + 中心 δ 势 | (57)–(64)：波函数尖点、基组收敛、区分两类误差 |
| `examples/ex3_three_level_degenerate.py` | 一个单态 + 一个二重简并块 | (110)–(124)：好基、二阶能量、一阶态矢、严格对角化及误差 |

三能级简并例题直接运行：

```bash
python examples/ex3_three_level_degenerate.py
# 精确打印 (E1, E2) = (0, -10), (-2, 1), (2, 9)
# 并保存 figures/ex3_three_level_degenerate.png
```

套路只有四步：**选基 → 写 $H_0$ 的对角元 → 写 $V$ 的矩阵 → `solve()`** 。以 δ 势阱为例，$\phi_n=\sqrt{2/L}\sin(n\pi x/L)$、$E_n^{(0)}=\pi^2n^2/2$、$V_{mn}=2\sin(m\pi/2)\sin(n\pi/2)$：

```python
N   = 2000
n   = np.arange(1, N + 1)
e0  = (np.pi * n) ** 2 / 2                       # H₀ 的本征值，ħ=M=L=1
sgn = np.where(n % 2 == 1, (-1.0) ** ((n - 1) // 2), 0.0)
V   = 2.0 * np.outer(sgn, sgn)                   # 秩 1
ser = PerturbationTheory(e0, V).solve(0, 8)
ser.dE[1], ser.dE[2]                             # (2.0, -0.20244)
```

三条容易踩的约定：

1. **λ 不要乘进 $V$** 。程序永远是 $H_0+\lambda V$。讲义式 (41) 的 $\Omega$ 就是 $\lambda$，$V$ 取 $S_x=\hbar\sigma_x/2$（即式 43 的矩阵）；若先把 $\Omega$ 乘进矩阵、再令 $\lambda=\Omega$，结果会多出 $\Omega$ 的幂次。
2. **整体平移不影响 $E^{(m)}$（$m\ge1$）** 。内置 `two_level(Δ)` 是 $\mathrm{diag}(0,\Delta)+v\sigma_x$，对应讲义的 $\delta=\Delta$、$\Omega=2v$，外加常数平移 $\Delta/2$；零点对齐后数值才逐位一致。
3. **基组截断要单独交代** 。δ 点势的非零 $V_{mn}$ 不随 $m$ 衰减，基态 $E^{(2)}$ 的误差以 $1/N$ 下降。$N=2000$ 时误差约为 $2.03\times10^{-4}$。取偶数 $N$，$2E^{(2)}(2N)-E^{(2)}(N)$ 可消去本例的基态二阶系数截断误差。

## 命令行

```bash
python -m qmpert --model x4 --N 120 --order 6 --lam 0.05
python -m qmpert --model x4 --order 20 --lam 0.2 --resum   # 追加 Padé / Borel–Padé
python -m qmpert --model x4 --N 20000 --sparse --order 40  # 稀疏后端
python -m qmpert --model displaced --N 120 --order 8
python -m qmpert --model two-level --exact --order 2 --show-state
python -m qmpert --model two-level --order 4 --lam 0.2 --plot figures/two_level_overview.png
python -m qmpert --model x4 --N 16 --exact --order 4  # 精确 Bender–Wu 系数
```

输出前几阶系数、给定 $\lambda$ 的截断能量、精确值（可对角化时）与误差。`--exact` 保持分数/符号算术；`--show-state` 同时打印逐阶范数归一化修正和给定 $\lambda$ 下整体归一化的态矢；`--plot` 直接保存系数与态概率图。

## 模块结构

| 文件 | 行数 | 职责 |
| --- | --- | --- |
| `qmpert/core.py` | — | 任意阶非简并递推、逐阶范数归一化：`PerturbationTheory` / `PerturbationSeries` |
| `qmpert/degenerate.py` | — | 二阶简并微扰：自动分块、好基、二阶能量与一阶态矢 |
| `qmpert/models.py` | 306 | 模型库：谐振子、$x^4$、位移振子、二能级、随机矩阵、稠密凸起、稀疏带状、sympy 精确版 |
| `qmpert/symmetry.py` | 224 | 选择定则：`ParitySymmetry` / `DeltaSymmetry` / `QuantumNumberSymmetry` / `CompositeSymmetry` |
| `qmpert/resum.py` | 322 | 重求和：`PadeApproximant` / `PadeTable` / `BorelPade` |
| `qmpert/viz.py` | — | 直接概览及 8 类报告图（收敛性、渐近性、对称性、稀疏结构、波函数、能级、重求和、规模） |
| `qmpert/report.py` | 787 | 自包含 HTML 报告生成器 |
| `qmpert/cli.py` | — | 命令行入口：精确系数、归一化态矢、直接出图 |
| `run_all.py` | 305 | 一键流水线：求解 → 出图 → 校验 → 出报告 |
| `validate.py` | — | 正确性校验，含解析式、对角化、跨路径一致性和二阶简并微扰 |
| `tests/` | — | pytest 回归，含归一化、CLI、例题与简并微扰 |

## 约定

读代码或对照教材前请先看这四条，否则很容易觉得公式“少了一项”。

1. **单位** ：$\hbar=m=\omega=1$，故 $x=(a+a^\dagger)/\sqrt2$，$H_0=\tfrac12(p^2+x^2)$。
2. **非简并**：目标态与其他态的零阶能隙须超过近简并容差。浮点谱的默认容差为 $10^{-10}(\max E^{(0)}-\min E^{(0)})$，精确谱检查精确相等。微扰的可靠性还取决于耦合与能隙的比值。
3. **中间归一化** ：$\langle n^{(0)}\lvert n(\lambda)\rangle\equiv1$，等价于 $\langle n\lvert n^{(m)}\rangle=0\ (m\ge1)$。代码里由 `c[m, n] = 0` 显式保证。
4. **两条递推** （$k\neq n$，$m\ge1$）：

$$
E_n^{(m)}=\langle n|V|n^{(m-1)}\rangle,\qquad
\langle k|n^{(m)}\rangle=\frac{-\langle k|V|n^{(m-1)}\rangle+\sum_{p=1}^{m-1}E_n^{(p)}\langle k|n^{(m-p)}\rangle}{E_k^{(0)}-E_n^{(0)}}
$$

能量式只有一项，是中间归一化的直接结果；若改用**按范数归一化** $\langle n(\lambda)\lvert n(\lambda)\rangle=1$，则要补上 $-\sum_{p=1}^{m-1}E_n^{(p)}\langle n\lvert n^{(m-p)}\rangle$。两种约定给出的 $E_n^{(m)}$ 逐阶相同。推导见 [`docs/algorithm.md`](docs/algorithm.md)。

## 复现

```bash
python run_all.py     # 扩展演示：figures/ + results.json + report.html
python validate.py    # 运行数值校验，写 validation.json
python -m pytest -q   # 回归测试
```

`run_all.py` 只在本次 `validate.py` 校验通过后生成报告。

## 文档

- [`docs/algorithm.md`](docs/algorithm.md) —— 从定态方程推导到迭代算法：两种归一化的转换、简并块一阶混合、矩阵化、伪代码、复杂度与三能级验收。
- `report.html` —— 扩展演示（四次非谐振子、稀疏后端、重求和等），运行 `run_all.py` 后用浏览器查看。

## 范围与限制

- **二阶简并求解的范围** 。输出能量到二阶、态矢到一阶。若有耦合的分支到二阶仍简并，默认报 `ResidualDegeneracyError`，可用 `solve(energy_only=True)` 获取二阶能量。浮点容差内合并的零阶能级以块代表能量计算。
- **截断误差随模型和耦合变化** 。例如四次非谐振子的级数为渐近展开。可和有限基组对角化对照，并验证所选截断阶。使用 `resum` 时也应独立检查结果。
- **数值精度** 。展开阶数增加时，系数增长与抵消可能超出浮点精度。`Fraction` 与 SymPy 可用于小维模型的精确系数，输入需使用整数、分数或精确符号表达式。
- **态矢系数是稠密数组** 。`coeffs` 形状 $(M+1)\times N$：$N=10^5$、40 阶约 33 MB。稀疏省的是 $V$ 的存储与乘法，不是这里。
- **精确对角化是 $O(N^3)$** 。`Model.exact_spectrum` 对稠密模型做全谱对角化，仅适合做基准；大规模请用稀疏路径的移位反演（`k` 取小）。
