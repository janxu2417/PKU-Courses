# 从物理公式到迭代算法

本文说明 `qmpert.core` 的递推是怎么来的：从定态方程出发，经过**逐阶比较 → 两次投影 → 向量化**三步，得到一段十几行的循环；并把每一步用到的约定写在明处。

读之前只需要知道一件事：**约定不同，公式长得不一样，但算出的数一样** 。第三节会专门处理这个最容易让人以为“代码漏项”的地方。

## 1 记号

| 符号                                             | 含义                                                              |
| ---------------------------------------------- | --------------------------------------------------------------- |
| $H(\lambda)=H_0+\lambda V$                     | 总哈密顿量，$\lambda$ 是形式耦合常数                                         |
| $\lvert n\rangle\equiv\lvert n^{(0)}\rangle$   | $H_0$ 的第 $n$ 个本征态，$H_0\lvert n\rangle=E_n^{(0)}\lvert n\rangle$ |
| $E_n^{(m)},\ \lvert n^{(m)}\rangle$            | 能量与态矢的第 $m$ 阶修正                                                 |
| $N$                                            | 截断基组维数（基矢 $\lvert 0\rangle\ldots\lvert N-1\rangle$）             |
| $M$                                            | 展开的最高阶                                                          |
| $V_{kn}\equiv\langle k\lvert V\rvert n\rangle$ | $V$ 在 $H_0$ 本征基下的矩阵元（**行指标是 bra** ）                              |

## 2 约定

### 2.1 单位与基

$\hbar=m=\omega=1$，于是 $x=(a+a^\dagger)/\sqrt2$，$H_0=\tfrac12(p^2+x^2)$，其本征值为 $\omega(n+\tfrac12)$。所有矩阵都在 $H_0$ 的本征基下表示，$V$ 由算符在该基下的矩阵给出（例如 $x^4$ 就是 $x$ 矩阵的四次连乘）。

基组必须截断到 $N$ 维。递推会用到 $\lvert n^{(m)}\rangle$ 在远处的分量，$N$ 要大到使最高阶修正在边界附近可忽略；`validate.py` 里 $N=120\ldots420$ 给出逐位相同的本征值，可作为选 $N$ 的参照。

### 2.2 展开与归一化

$$
E_n(\lambda)=\sum_{m\ge0}\lambda^mE_n^{(m)},\qquad
\lvert n(\lambda)\rangle=\sum_{m\ge0}\lambda^m\lvert n^{(m)}\rangle
$$

$\lvert n(\lambda)\rangle$ 的**整体标量因子**没有物理意义，必须由归一化条件固定。两种常见选择：

| 约定          | 条件                                             | 后果                                                     |
| ----------- | ---------------------------------------------- | ------------------------------------------------------ |
| 中间归一化（递推采用） | $\langle n^{(0)}\lvert n(\lambda)\rangle=1$    | $\langle n\lvert n^{(m)}\rangle=0$ 对一切 $m\ge1$；能量式只剩一项 |
| 按范数归一化      | $\langle n(\lambda)\lvert n(\lambda)\rangle=1$ | $\langle n\lvert n^{(m)}\rangle\neq0$（偶数阶起）；能量式要带求和项   |

中间归一化的等价说法：每一阶修正都与未微扰态**正交** 。代码里就是 `c[m, n] = 0`。

### 2.3 为什么选中间归一化

1. **能量式最短** ：$E_n^{(m)}=\langle n\lvert V\rvert n^{(m-1)}\rangle$，后面第四节整段算法都建立在这一行上。
2. **不需要逐阶重新归一化** ：每阶只需解一个线性方程，不必在阶与阶之间做 Gram–Schmidt。
3. **与教材一致** ：绝大多数量子力学教材的 $E^{(2)}=\sum_{k\ne n}\lvert V_{kn}\rvert^2/(E_n-E_k)$ 就是这套约定下的结果。

## 3 从定态方程到逐阶方程

### 3.1 代入

把两个展开式代进 $H(\lambda)\lvert n(\lambda)\rangle=E_n(\lambda)\lvert n(\lambda)\rangle$，比较 $\lambda^m$ 的系数：

$$
\underbrace{H_0\lvert n^{(m)}\rangle+V\lvert n^{(m-1)}\rangle}_{\text{左端}}
=\underbrace{\sum_{p=0}^{m}E_n^{(p)}\lvert n^{(m-p)}\rangle}_{\text{右端}}
$$

左端只有两项（$\lambda$ 一次只能从 $H_0$ 或 $\lambda V$ 里来），右端是卷积。

### 3.2 整理成逐阶方程

$m=0$ 给出 $H_0\lvert n^{(0)}\rangle=E_n^{(0)}\lvert n^{(0)}\rangle$，这是定义式，不含新信息。  
对 $m\ge1$，把右端的 $p=0$ 项 $E_n^{(0)}\lvert n^{(m)}\rangle$ 挪到左边：

$$
\boxed{(H_0-E_n^{(0)})\lvert n^{(m)}\rangle=-V\lvert n^{(m-1)}\rangle+\sum_{p=1}^{m}E_n^{(p)}\lvert n^{(m-p)}\rangle}
\tag{3.1}
$$

**这一步是全篇的枢纽** ：左端算子 $H_0-E_n^{(0)}$ 在 $\lvert n\rangle$ 上为零、在其他基矢上给出能隙 $E_k^{(0)}-E_n^{(0)}$。于是“投影到 $\langle n\rvert$”用来定能量，“投影到 $\langle k\rvert$”用来定态矢，一次方程拆成两件事。

### 3.3 投影到 $\langle n\rvert$：能量

左乘 $\langle n^{(0)}\rvert$。左端

$$
\langle n\lvert(H_0-E_n^{(0)})\rvert n^{(m)}\rangle=(E_n^{(0)}-E_n^{(0)})\langle n\lvert n^{(m)}\rangle=0
$$

右端逐项看：

| $p$      | 因子 $\langle n\lvert n^{(m-p)}\rangle$ | 贡献          |
| -------- | ------------------------------------- | ----------- |
| $p=m$    | $\langle n\lvert n^{(0)}\rangle=1$    | $E_n^{(m)}$ |
| $p=m-1$  | $\langle n\lvert n^{(1)}\rangle=0$    | 0           |
| $\cdots$ | $\cdots$                              | 0           |
| $p=1$    | $\langle n\lvert n^{(m-1)}\rangle=0$  | 0           |

中间归一化把 $p<m$ 的项**全部**按定义消掉（不是因为它们在方程层面被删了），只剩 $p=m$：

$$
0=-\langle n\lvert V\rvert n^{(m-1)}\rangle+E_n^{(m)}
\quad\Longrightarrow\quad
\boxed{E_n^{(m)}=\langle n\lvert V\rvert n^{(m-1)}\rangle}
\tag{3.2}
$$

这就是本项目用的能量式。**它依赖 2.2 的约定** ：换成按范数归一化，这串求和不会消失。

### 3.4 投影到 $\langle k\rvert$：态矢

左乘 $\langle k^{(0)}\rvert$（$k\ne n$）。左端给出 $(E_k^{(0)}-E_n^{(0)})\langle k\lvert n^{(m)}\rangle$；右端 $p=m$ 项是 $E_n^{(m)}\langle k\lvert n^{(0)}\rangle=0$（因为 $k\ne n$），所以求和只能到 $m-1$：

$$
\boxed{\langle k\lvert n^{(m)}\rangle=\frac{-\langle k\lvert V\rvert n^{(m-1)}\rangle+\sum_{p=1}^{m-1}E_n^{(p)}\langle k\lvert n^{(m-p)}\rangle}{E_k^{(0)}-E_n^{(0)}}}
\tag{3.3}
$$

同一个求和，在 (3.2) 里被归一化条件清零、在 (3.3) 里必须保留——因为 $k\ne n$ 时 $\langle k\lvert n^{(m-p)}\rangle$ 一般不为零。它们是**低阶能量对高阶态矢的反馈** ，也是这条递推“自洽”二字的由来。

### 3.5 与按范数归一化的对照

若用 $\langle n(\lambda)\lvert n(\lambda)\rangle=1$，范数条件逐阶给出

$$
\langle n\lvert n^{(m)}\rangle=-\frac12\sum_{a=1}^{m-1}\langle n^{(a)}\lvert n^{(m-a)}\rangle
\quad\Longrightarrow\quad
\langle n\lvert n^{(2)}\rangle=-\frac12\sum_{k\ne n}\frac{\lvert V_{kn}\rvert^2}{(E_n-E_k)^2}\ne0
$$

此时能量式变成

$$
E_n^{(m)}=\langle n\lvert V\rvert n^{(m-1)}\rangle-\sum_{p=1}^{m-1}E_n^{(p)}\langle n\lvert n^{(m-p)}\rangle
\tag{3.4}
$$

两式的差别全部来自 $\langle n\lvert n^{(m)}\rangle$。**数值上完全等价** ：$E_n(\lambda)$ 是本征值，与态矢怎么归一化无关，而 $\lambda=0$ 处的 Taylor 系数唯一。注意 $\langle n\lvert n^{(1)}\rangle=0$，所以二阶上两式碰巧长得一样，差异从三阶才显现——这也是为什么很多人只验到二阶时看不出问题。

如果只记一条：**先看约定，再抄公式** 。

### 3.6 代码怎样从中间归一化转到逐阶范数归一化

递推仍使用中间归一化系数 $\mathbf c_m$（`coeffs[m]`），算完后再构造范数归一化系数 $\mathbf u_m$（`normalized_coeffs()[m]`）。令

$$
\lvert n_N\rangle=A(\lambda)\lvert n_I\rangle,\qquad
A(\lambda)=1+\sum_{m\ge1}\lambda^m a_m,
$$

其中 $A$ 取实数，因而不改变中间归一化态的相位约定。展开并比较系数得

$$
\boxed{\ \mathbf u_m=\mathbf c_m+\sum_{r=1}^{m}a_r\mathbf c_{m-r}\ },\qquad
a_m=\langle n\lvert n^{(m)}_N\rangle .
$$

逐阶范数条件 $\sum_{k=0}^m\langle n_N^{(k)}\lvert n_N^{(m-k)}\rangle=\delta_{m0}$ 给出 $a_1=0$，以及

$$
\boxed{\ a_m=-\frac12\sum_{k=1}^{m-1}
\langle n_N^{(k)}\lvert n_N^{(m-k)}\rangle\ }\qquad(m\ge2).
$$

右端的求和必为实数；数值后端只取实部以去掉舍入产生的虚部，SymPy 后端则用 `as_real_imag()` 保持精确算术。特别是

$$
\mathbf u_1=\mathbf c_1,\qquad
a_2=-\frac12\langle\mathbf c_1\lvert\mathbf c_1\rangle
=-\frac12\sum_{k\ne n}\frac{\lvert V_{kn}\rvert^2}{(E_n-E_k)^2}.
$$

这与讲义式 (24) 一致。两种归一化只相差标量因子 $A(\lambda)$，所以**能量系数不变** 。还要注意两个不同操作：

- `normalized_coeffs()`：返回逐阶系数 $\mathbf u_m$，用于对照讲义公式；
- `normalized_state(λ, order)`：先把中间归一化级数截断，再对有限项向量的整体范数归一化，用于数值和作图。它一般含有超出 `order` 的归一化修正，不能当作同一阶的 $\mathbf u_m$。

## 4 从逐阶方程到矩阵迭代

### 4.1 向量化

把第 $m$ 阶修正写成一个 $N$ 维列向量

$$
\mathbf{c}^{(m)}\in\mathbb{C}^N,\qquad c_k^{(m)}=\langle k\lvert n^{(m)}\rangle
$$

再记 $\mathbf{e}_n$ 为第 $n$ 个单位向量、$\mathbf{g}$ 为能隙向量：

$$
g_k=E_k^{(0)}-E_n^{(0)}\quad(k\ne n),\qquad g_n=1
$$

$g_n=1$ 是占位值。逐元素除法完成后，程序把目标态分量显式设为精确的零。

两个关键观察：

1. 矩阵–向量乘给出全部矩阵元：$(V\mathbf{c}^{(m-1)})_k=\sum_jV_{kj}c_j^{(m-1)}=\langle k\lvert V\rvert n^{(m-1)}\rangle$。取第 $n$ 个分量就是能量 (3.2)。
2. (3.3) 的分子对每个 $k$ 是同一个结构，因此可以整体写成向量式。

于是 (3.2)(3.3) 合成一条：

$$
\boxed{\ \mathbf{c}^{(m)}=(I-\mathbf{e}_n\mathbf{e}_n^\dagger)
\left[\Big(-V\mathbf{c}^{(m-1)}+\sum_{p=1}^{m-1}E_n^{(p)}\mathbf{c}^{(m-p)}\Big)\oslash\mathbf{g}\right]\ }
\tag{4.1}
$$

其中 $\oslash$ 是逐元素除法。**这就是“物理公式 → 迭代算法”的全部内容** ：一个矩阵–向量乘，加上 $m-1$ 个已知向量的线性组合。

### 4.2 伪代码

```text
输入: e0[N], V[N][N], 目标态 n, 最高阶 M
输出: dE[0..M], c[0..M][N]

c[0] ← 0;  c[0][n] ← 1          # |n^(0)> = |n>
dE[0] ← 0
g ← e0 - e0[n];  g[n] ← 1       # 分母占位

for m = 1 .. M:
    v      = V @ c[m-1]         # 唯一一次矩阵–向量乘
    dE[m]  = v[n]               # E^(m) = <n|V|n^(m-1)>
    rhs    = -v
    for p = 1 .. m-1:
        rhs += dE[p] * c[m-p]   # 低阶能量反馈
    c[m]   = rhs / g            # 逐元素
    c[m][n] = 0                 # 中间归一化
```

### 4.3 与 `core.py` 的对应

| 伪代码                        | `core.py` | 说明                          |
| -------------------------- | --------- | --------------------------- |
| `g ← e0 - e0[n]; g[n] ← 1` | `gaps`   | 能隙向量与目标态占位值 |
| `v = V @ c[m-1]`           | `vc`     | 稠密走 BLAS，稀疏走 CSR，精确后端走显式循环 |
| `dE[m] = v[n]`             | `dE[m]`  | 式 (3.2) |
| `rhs += dE[p] * c[m-p]`    | `rhs`    | $p=1\ldots m-1$，式 (3.3) 的求和 |
| `c[m] = rhs / g`           | `c[m]`   | 式 (4.1)，最后设 `c[m, n] = 0` |

`solve()` 里没有别的物理，剩下的都是 dtype 处理与边界检查。

### 4.4 复杂度

| 项        | 稠密             | 稀疏                           |
| -------- | -------------- | ---------------------------- |
| 每阶矩阵–向量乘 | $O(N^2)$       | $O(\mathrm{nnz})$            |
| 每阶反馈求和   | $O(mN)$        | $O(mN)$                      |
| 总量       | $O(MN^2+M^2N)$ | $O(M\cdot\mathrm{nnz}+M^2N)$ |
| 内存       | $O(MN)$        | $O(MN)$                      |

**一处需要说清楚的地方** ：反馈求和 $\sum_pE^{(p)}\mathbf{c}^{(m-p)}$ 是稠密向量运算，总代价 $M^2N/2$，与 $\mathrm{nnz}$ 无关。$N=10^5$、$M=40$ 时它约 $8\times10^7$ 次乘加，和 $M\cdot\mathrm{nnz}\approx2\times10^7$ 同量级甚至更大——报告里“单阶 $O(\mathrm{nnz})$”说的是 matvec 那一项，不是全部。两者都正比于 $N$，所以稀疏路径整体仍是 $O(N)$ 量级，结论不变，但常数不能忽略。

### 4.5 为什么能“任意阶”

递推用第 $m-1$ 阶态矢做一次矩阵–向量乘，并将 $1$ 到 $m-1$ 阶的已知系数组合成反馈项。计算顺次推进，各阶使用同一组公式。

$x^4$ 的级数系数快速增长。双精度计算要留意溢出与抵消误差，精确系数可用 `Fraction` 或 SymPy 后端计算。给定 $\lambda$ 后，截断阶数应结合级数收敛性和基组验证。

## 5 选择定则插在哪一步

**不进入循环** 。选择定则只在构造 $V$ 时生效，递推本身对“$V$ 有没有对称性”一无所知：

| 用途   | 时机         | 效果                                                              |
| ---- | ---------- | --------------------------------------------------------------- |
| 校验   | 求解前        | 统计禁戒位置的最大元素，超过容差说明假设的对称性不成立，应报错而不是继续算                           |
| 掩码   | 求解前        | 把禁戒元清零，矩阵严格分块；稀疏时只保留 nnz 中允许的部分，$O(\mathrm{nnz})$               |
| 先验判零 | 求解前（与数值无关） | `predicted_zero_orders` 给出“必然为零”的阶，供和 `measured_zero_orders` 对照 |

两点注意：

1. 数值生成的 $V$（例如实空间投影得到的高斯凸起）在禁戒位置有 $10^{-16}$ 级求积噪声。掩码把它们清零，**物理不变** ，但矩阵变成严格分块——这是 `gaussian_bump` 演示的场景。
2. 若规则本身不成立（禁戒位置有真实大小的元），掩码会**改变物理** 。所以 `PerturbationTheory` 会先算 `violation` 并记在 `diagnostics()` 里；`enforce_symmetry` 只是“在自洽前提下利用结构”，不是“用对称性替代计算”。

## 6 数值细节与陷阱

| # | 事项                                    | 处理方式                                                                                    |
| - | ------------------------------------- | --------------------------------------------------------------------------------------- |
| 1 | 分母 $E_k-E_n$ 接近零 | 浮点容差默认为 $10^{-10}(\max E^{(0)}-\min E^{(0)})$，精确谱用精确相等 |
| 2 | 对角分量 $\langle n\lvert n^{(m)}\rangle$ | `g[n]=1` 占位，目标态分量在除法后显式置零 |
| 3 | 精确运算 | 用整数、`Fraction` 或 SymPy 精确数值构建 `object` 数组，整数能隙转成 `Fraction` |
| 4 | 判零诊断 | 精确后端比较零值，浮点默认按各阶能量内积中各项绝对值之和定标，选择定则可独立给出严格零阶 |
| 5 | 级数是渐近的                                | 固定 $\lambda$ 存在最佳截断阶 $m^\ast\approx1/(3\lambda)$；超过只会更差，要更大的 $\lambda$ 请用 `resum`       |
| 6 | 基组截断                                  | $\mathbf{c}^{(m)}$ 的支撑随 $m$ 变宽（$x^4$ 每阶最远 $\Delta n=4$），取 $N\gtrsim n+4M$ 才不会被边界反射      |
| 7 | 稀疏求精确谱                                | 移位反演对带状矩阵接近 $O(N)$，但移位量取在基态附近；取太低会让 Lanczos 不收敛                                         |

## 7 公式与校验的对应

| 公式 / 结论                                      | 校验项（`validate.py`）                                | 实测            |
| -------------------------------------------- | ------------------------------------------------- | ------------- |
| (3.2) 一阶：$E_n^{(1)}=\tfrac34(2n^2+2n+1)$     | `x^4 E_n^(1) vs 解析`                               | 3e−16 … 5e−15 |
| (3.2) 二阶：$E_n^{(2)}=-(34n^3+51n^2+59n+21)/8$ | `x^4 E_n^(2) vs 解析`                               | 0 … 6e−14     |
| (3.1)+(3.2) 全阶                               | `随机模型 8 阶 vs 精确`                                  | 8e−15         |
| 中间归一化 $\langle n\lvert n^{(m)}\rangle=0$     | 单测 `test_state_vector_intermediate_normalization` | 通过            |
| 精确有理系数                                       | `E0^(1..6)` 与 Bender–Wu 对照                        | 逐位一致          |
| 二能级闭式                                        | `偶数阶系数 vs 闭式`                                     | 0             |

## 8 三十秒自检

不必等程序：一阶系数可以手算。$x^4=\tfrac14(a+a^\dagger)^4$，展开成 16 项后，只有 $a$ 与 $a^\dagger$ 数目相等的 6 项在 $\lvert n\rangle$ 上非零；逐项用 $\langle a^\dagger a\rangle=n$ 与 $\langle aa^\dagger\rangle=n+1$ 求和得 $\langle n\lvert(a+a^\dagger)^4\rvert n\rangle=6n^2+6n+3$，于是

$$
E_n^{(1)}=\frac14(6n^2+6n+3)=\frac34(2n^2+2n+1)
$$

注意这里**不能**用 $\langle x^4\rangle=3\langle x^2\rangle^2$：那条 Wick 恒等式只对高斯态（$n=0$）成立，$n=1$ 时 $\langle x^4\rangle=15/4$ 而 $3\langle x^2\rangle^2=27/4$，差得很远。

比一比：$n=0$ 给 $3/4$，$n=1$ 给 $15/4$，与程序输出一致。二阶就没有这种一行技巧了，但可以拿 $E_0^{(2)}=-21/8$、$E_0^{(3)}=333/16$、$E_0^{(4)}=-30885/128$ 这三个 Bender–Wu 系数当基准——它们对任何实现都是硬约束。

位移振子更适合练手：奇宇称微扰 ⇒ 奇阶为零，且级数在二阶精确终止（$E_0=\tfrac12-F^2/2$），手算到二阶就能和解析解对上。

## 9 二阶简并微扰与三能级验收

`degenerate.py` 按讲义计算二阶能量和一阶态矢。先按 $H_0$ 的本征值分块，再对角化每个块 $P$ 内的 $PVP$，得到一阶能量 $v_\alpha$。若一阶仍简并，须按下文在残余子空间内对角化二阶有效矩阵以选定好基。记最终好基下的微扰矩阵元为 $\widetilde V_{\beta\alpha}$，则块外系数与二阶能量为

$$
c_{\beta\alpha}^{(1)}=\frac{\widetilde V_{\beta\alpha}}{E_P-E_\beta}
\quad(\beta\notin P),\qquad
E_\alpha^{(2)}=\sum_{\beta\notin P}
\frac{\lvert\widetilde V_{\beta\alpha}\rvert^2}{E_P-E_\beta}.
$$

同一块内不同好基之间的一阶混合不能由一阶方程的投影直接定出，要继续用二阶方程；当 $v_\beta\ne v_\alpha$ 时，

$$
\boxed{\ c_{\beta\alpha}^{(1)}=
\sum_{\gamma\notin P}
\frac{\widetilde V_{\beta\gamma}\widetilde V_{\gamma\alpha}}
{(v_\beta-v_\alpha)(E_\gamma-E_P)}\ }\qquad
(\beta\in P,\ \beta\ne\alpha).
$$

若 $PVP$ 仍有重复本征值，记 $S$ 为一阶能量 $v_\alpha$ 对应的子空间，$T=P-S$。令 $Q=I-P$，$R=(E_P-H_{0,Q})^{-1}$，$B=QVP$，$X=RB$，$W_2=B^\dagger X$。二阶方程投影到 $P$ 内的分量为

$$
(v_\beta-v_\alpha)c_{\beta\alpha}^{(1)}
  +(W_2)_{\beta\alpha}-E_\alpha^{(2)}\delta_{\beta\alpha}=0
\qquad(\beta\in P).
$$

对 $\beta\in S$，$v_\beta-v_\alpha=0$，所以二阶投影只能给出 $W_{2,SS}$ 的本征方程。其本征向量选定零阶好基 $\lvert\alpha\rangle$，本征值是 $E_\alpha^{(2)}$。对 $T$ 的投影则给出

$$
T\lvert\alpha^{(1)}\rangle
  = \bigl(v_\alpha I_T-(PVP)_{TT}\bigr)^{-1}W_{2,TS}\lvert\alpha\rangle .
$$

对 $Q$ 的二阶投影可写成

$$
Q\lvert\alpha^{(2)}\rangle
  = X P\lvert\alpha^{(1)}\rangle
  + R(V_{QQ}-v_\alpha I_Q)X\lvert\alpha\rangle .
$$

把这两个结果代入三阶方程并投影到 $S$，得到

$$
(W_{2,SS}-E_\alpha^{(2)}I_S)S\lvert\alpha^{(1)}\rangle
  + K_3\lvert\alpha\rangle = E_\alpha^{(3)}\lvert\alpha\rangle ,
$$

$$
K_3=X_S^\dagger(V_{QQ}-v_\alpha I_Q)X_S
  +W_{2,ST}\bigl(v_\alpha I_T-(PVP)_{TT}\bigr)^{-1}W_{2,TS}.
$$

第一项来自 $S\to Q\to Q\to S$，其中 $-v_\alpha I_Q$ 来自一阶能量修正。第二项来自 $S\to Q\to T\to Q\to S$，$T$ 中间态带入一阶能隙分母。若二阶已劈裂，沿 $\beta\in S$ 且 $\beta\ne\alpha$ 的系数为

$$
c_{\beta\alpha}^{(1)}
  = \frac{(K_3)_{\beta\alpha}}{E_\alpha^{(2)}-E_\beta^{(2)}} .
$$

三阶矩阵元的量级为 $\lambda^3$，二阶能隙的量级为 $\lambda^2$，两者相除给出 $\lambda c_{\beta\alpha}^{(1)}$。因此求一阶态矢需要推进到三阶投影，能量输出仍截在二阶。[`_state_corrections()`](../qmpert/degenerate.py#L223) 先把 $T$ 中的混合写入 `within`，再用 `Xg`、`C` 和 `within` 构造 `K3`，最后用 `second_order[alpha] - second_order[beta]` 作分母。

二阶之后仍简并且与块外有耦合时，`solve(energy_only=True)` 返回二阶能量和零阶好基，`c1` 为 `None`。默认同时求态矢，此时会报 `ResidualDegeneracyError`。输出的态矢系数始终转回输入的原基。

三能级例题以 $E_0$ 为能量单位，$\lambda=V_0/E_0$，$V$ 为去除公因子 $V_0$ 后的矩阵。

$$
e^{(0)}=(0,1,1),\qquad
V=\begin{pmatrix}
0&\sqrt2&2\sqrt2\\
\sqrt2&0&2\\
2\sqrt2&2&0
\end{pmatrix}.
$$

程序精确输出三条分支的 $(E^{(1)},E^{(2)})=(0,-10),(-2,1),(2,9)$；二重简并块的好基为 $(\lvert2\rangle\mp\lvert3\rangle)/\sqrt2$，一阶原基系数依次为

$$
(0,-\sqrt2,-2\sqrt2),\qquad
\left(-1,\frac3{4\sqrt2},\frac3{4\sqrt2}\right),\qquad
\left(3,-\frac3{4\sqrt2},\frac3{4\sqrt2}\right).
$$

这些量逐项对应讲义式 (110)–(124)，并由 SymPy 精确测试和小 $\lambda$ 严格对角化共同验证。
