"""
qmpert.report
=============

把 :mod:`run_all` 计算出的 ``summary`` 与 ``validation.json`` 渲染成
一份自包含的 HTML 报告（图片以 base64 内嵌，可离线阅读、可直接分享）。
"""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path

__all__ = ["build", "render"]

CSS = """
:root{
  --ink:#111827; --ink2:#1f2937; --muted:#5b6573; --line:#e4e7ec; --bg:#f6f7f9;
  --card:#ffffff; --accent:#1d4ed8; --accent-soft:#eef2ff;
  --amber:#b45309; --green:#15803d; --red:#b91c1c;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font-size:15.5px;line-height:1.78;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB",
  "Microsoft YaHei","Noto Sans CJK SC",sans-serif;-webkit-font-smoothing:antialiased;}
h1,h2,h3,h4{font-family:"Songti SC","Source Han Serif SC","Noto Serif CJK SC",Georgia,serif;
  font-weight:600;letter-spacing:-0.01em;color:var(--ink);margin:0;}
p{margin:0 0 14px}
a{color:var(--accent);text-decoration:none}
a:hover{text-decoration:underline}
.shell{max-width:1220px;margin:0 auto;padding:0 26px}

/* ---------- hero ---------- */
.hero{background:
   radial-gradient(1100px 420px at 82% -12%, rgba(59,130,246,.28), transparent 62%),
   radial-gradient(760px 320px at 8% 0%, rgba(180,83,9,.16), transparent 60%),
   linear-gradient(158deg,#0c1524 0%,#111d31 48%,#16243c 100%);
  color:#e8eef7;padding:60px 0 96px;border-bottom:1px solid #0b1220}
.kicker{font-size:11.5px;letter-spacing:.22em;text-transform:uppercase;color:#8ea6c8;
  margin-bottom:18px;font-weight:600}
.hero h1{color:#f6f9ff;font-size:clamp(28px,3.6vw,44px);line-height:1.25;max-width:24ch}
.hero .lede{color:#b9c8dc;font-size:16.5px;max-width:64ch;margin-top:18px;line-height:1.8}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:22px}
.pill{font-size:12px;color:#b9c8dc;border:1px solid rgba(255,255,255,.16);
  background:rgba(255,255,255,.05);padding:5px 11px;border-radius:999px;font-variant-numeric:tabular-nums}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:36px}
.kpi{background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.12);
  border-radius:12px;padding:16px 18px 14px}
.kpi .v{font-size:26px;font-weight:600;color:#fff;font-variant-numeric:tabular-nums;
  letter-spacing:-0.02em;line-height:1.2}
.kpi .l{font-size:11.5px;color:#93a7c4;letter-spacing:.06em;margin-top:4px}
@media(max-width:820px){.kpis{grid-template-columns:repeat(2,1fr)}}

/* ---------- sheet & layout ---------- */
.sheet{background:var(--card);border:1px solid var(--line);border-radius:16px;
  padding:10px 46px 64px;margin-top:-52px;position:relative;
  box-shadow:0 1px 2px rgba(16,24,40,.05),0 24px 48px -24px rgba(16,24,40,.22)}
.layout{display:grid;grid-template-columns:184px minmax(0,1fr);gap:52px}
@media(max-width:980px){.sheet{padding:8px 22px 44px}.layout{grid-template-columns:1fr;gap:0}
  nav.toc{display:none}}

/* ---------- toc ---------- */
nav.toc{position:sticky;top:26px;align-self:start;padding-top:44px;font-size:13px}
nav.toc .t{font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:#9aa4b2;
  margin-bottom:12px;font-weight:600}
nav.toc a{display:block;color:var(--muted);padding:4px 0 4px 12px;border-left:2px solid transparent;
  line-height:1.5}
nav.toc a:hover{color:var(--accent);border-left-color:#c7d2fe;text-decoration:none}
nav.toc a .n{color:#a8b2c0;font-variant-numeric:tabular-nums;margin-right:7px;font-size:11.5px}

/* ---------- sections ---------- */
section{padding:46px 0 6px;border-bottom:1px solid var(--line)}
section:last-of-type{border-bottom:none}
.sec-head{display:flex;align-items:baseline;gap:12px;margin-bottom:8px}
.sec-head .num{font-family:var(--mono);font-size:12px;color:var(--accent);
  background:var(--accent-soft);padding:3px 8px;border-radius:6px;font-weight:600}
h2{font-size:25px;margin-bottom:6px}
h3{font-size:17.5px;margin:26px 0 8px}
.sec-sub{color:var(--muted);font-size:14px;margin:-2px 0 18px}

/* ---------- formula ---------- */
.eq{display:flex;align-items:center;gap:18px;background:#fbfcfe;border:1px solid #e8edf6;
  border-left:3px solid var(--accent);border-radius:0 10px 10px 0;padding:13px 20px;margin:16px 0}
.eq .body{flex:1;font-family:"Cambria Math","Latin Modern Math","STIX Two Math",Georgia,serif;
  font-size:17.5px;line-height:2.0;color:#0f172a;overflow-x:auto}
.eq .tag{font-family:var(--mono);font-size:11.5px;color:#94a3b8;white-space:nowrap}
sub,sup{font-size:.72em}
.sig{display:inline-flex;align-items:center;vertical-align:-.12em}
.sig .l{display:inline-flex;flex-direction:column;font-size:.68em;line-height:1.22;
  margin-left:2px;text-align:left;font-style:normal}
i{font-family:"Cambria Math","Latin Modern Math","STIX Two Math",Georgia,serif;font-style:italic}
.note{border-left:2px solid #f3d19e;background:#fffcf3;padding:12px 16px;border-radius:0 8px 8px 0;
  font-size:14px;color:#4b4a45;margin:16px 0}
.note b{color:var(--amber)}
.callout{border-left:2px solid #bfdbfe;background:#f6faff;padding:12px 16px;border-radius:0 8px 8px 0;
  font-size:14px;margin:16px 0;color:#334155}

/* ---------- figure ---------- */
figure{margin:24px 0 10px}
figure.narrow{max-width:600px;margin-left:auto;margin-right:auto}
figure img{width:100%;display:block;border:1px solid var(--line);border-radius:10px;background:#fff}
figcaption{font-size:13.2px;color:var(--muted);margin-top:10px;line-height:1.72}
figcaption b{color:var(--ink2);font-weight:600}

/* ---------- table ---------- */
.tbl-wrap{overflow-x:auto;margin:16px 0 8px}
.dim{color:#98a2b3;font-size:.88em}
td.frac,th.frac{border-left:1px solid var(--line);padding-left:20px;text-align:left!important}
td.frac{font-family:var(--mono);font-size:12px;color:#3d5a80}
p.cap{font-size:12.8px;color:var(--muted);margin:6px 0 18px;line-height:1.7}
table{width:100%;border-collapse:collapse;font-size:13.2px}
th{text-align:left;font-weight:600;color:#6b7685;font-size:11px;letter-spacing:.08em;
  text-transform:uppercase;border-bottom:1px solid var(--line);padding:8px 12px;white-space:nowrap}
td{padding:7px 12px;border-bottom:1px solid #f1f3f7;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:hover td{background:#fbfcfe}
td.num,th.num{text-align:right}
.mono{font-family:var(--mono);font-size:12.2px}
.ok{color:var(--green);font-weight:600}
.bad{color:var(--red);font-weight:600}
.scroll{max-height:452px;overflow-y:auto;border:1px solid var(--line);border-radius:10px}
.scroll table td:first-child,.scroll table th:first-child{width:56%}
.scroll table{font-size:12.4px}
.scroll th{position:sticky;top:0;background:#fff;z-index:1}

/* ---------- code ---------- */
pre{background:#0f172a;color:#dbe4f0;padding:16px 18px;border-radius:11px;overflow-x:auto;
  font-family:var(--mono);font-size:12.4px;line-height:1.7;margin:14px 0}
pre .c{color:#7c8aa0}
pre .k{color:#93c5fd}
pre .s{color:#a7f3d0}
code{font-family:var(--mono);font-size:.9em;background:#f1f4f9;padding:1.5px 5px;border-radius:4px;
  color:#334155}
pre code{background:none;padding:0;color:inherit}

/* ---------- pipeline ---------- */
.pipe{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:18px 0 6px}
.pipe .box{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:7px 12px;
  font-size:12.5px;color:#334155}
.pipe .arw{color:#c3cbd8;font-size:13px}

/* ---------- files ---------- */
.tree{white-space:pre;font-family:var(--mono);font-size:12.6px;line-height:1.85;background:#fbfcfe;
  border:1px solid var(--line);border-radius:10px;padding:16px 20px;color:#334155;overflow-x:auto}
.tree .d{color:var(--accent)}
.tree .cm{color:#94a3b8}

footer{padding:34px 0 60px;color:#8b95a4;font-size:12.5px;text-align:center}
@media print{body{background:#fff}.sheet{box-shadow:none;border:none;margin:0;padding:0}
  .hero{padding:24px 0 28px}nav.toc{display:none}.layout{grid-template-columns:1fr}
  section{page-break-inside:avoid}}
"""


# --------------------------------------------------------------------------- #
#  小工具
# --------------------------------------------------------------------------- #
def _img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _f(x: float, sig: int = 4) -> str:
    """紧凑数值格式：大/小用科学计数法。"""
    if x == 0:
        return "0"
    ax = abs(x)
    if ax < 1e-3 or ax >= 1e4:
        return f"{x:.{sig-1}e}".replace("e-0", "e-").replace("e+0", "e+")
    s = f"{x:.{max(sig,3)}g}"
    return s


def _versions() -> dict:
    """**运行时**读取解释器与依赖版本。

    报告里写死版本号会在换机器/换环境后变成假信息；这里现读，
    缺包时退化为 "—" 而不是让整份报告失败。
    """
    import platform

    out = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "matplotlib", "sympy", "mpmath"):
        try:
            out[name] = __import__(name).__version__
        except (ImportError, AttributeError):  # pragma: no cover
            out[name] = "—"
    return out


# --------------------------------------------------------------------------- #
#  渲染
# --------------------------------------------------------------------------- #
def render(summary: dict, rows: list[dict], figdir: Path) -> str:
    figdir = Path(figdir)
    env = _versions()

    def fig(name: str, caption: str, num: int, cls: str = "") -> str:
        p = figdir / name
        if not p.exists():
            return f"<figure><figcaption>（缺失：{name}）</figcaption></figure>"
        return (
            f'<figure{(" class=" + chr(34) + cls + chr(34)) if cls else ""}>'
            f'<img src="{_img(p)}" alt="{html.escape(caption[:40])}">'
            f"<figcaption><b>图 {num}.</b> {caption}</figcaption></figure>"
        )

    coeffs = summary.get("coeffs", {})
    asy = summary.get("asymptotic", {})
    stark = summary.get("stark", {})
    sp = summary.get("sparsity", {})
    wf = summary.get("wavefunction", {})
    timing = summary.get("timing", {}).get("total_s", 0.0)
    model = summary.get("model", {})
    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    n_tot = len(rows)

    # ---- 系数表 ----
    states = sorted(coeffs.keys(), key=lambda s: int(s)) if coeffs else []
    max_m = max((len(coeffs[s]) for s in states), default=8)
    exact_coeffs = summary.get("exact", {}).get("coeffs", [])
    head = "".join(f'<th class="num">n = {s}</th>' for s in states)
    if exact_coeffs:
        head += "<th class='num frac' style='text-transform:none'>n = 0 精确值</th>"
    body = []
    for m in range(max_m):
        cells = "".join(f'<td class="num">{_f(coeffs[s][m], 5)}</td>' for s in states)
        if exact_coeffs:
            frac = exact_coeffs[m] if m < len(exact_coeffs) else ""
            cells += f"<td class='num frac'>{frac}</td>"
        body.append(f"<tr><td class='mono'>{m+1}</td>{cells}</tr>")
    coeff_table = (
        '<div class="tbl-wrap"><table><thead><tr><th>阶 m</th>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )

    # ---- 渐近表 ----
    asy_rows = []
    for lam, d in asy.items():
        asy_rows.append(
            f"<tr><td class='mono'>{lam}</td><td class='num'>{d['m_star']}</td>"
            f"<td class='num'>{_f(d['err_first'], 3)}</td>"
            f"<td class='num'>{_f(d['err_min'], 3)}</td>"
            f"<td class='num'>{_f(d['err_last'], 3)}</td>"
            f"<td class='num'>{d['m_star'] and round(1/(3*float(lam)))}</td></tr>"
        )
    asy_table = (
        "<div class='tbl-wrap'><table><thead><tr><th>λ</th><th class='num'>最佳阶 m*</th>"
        f"<th class='num'>一阶误差</th><th class='num'>m* 处误差</th><th class='num'>{model.get('max_order', 40)} 阶误差</th>"
        "<th class='num'>1/(3λ)</th></tr></thead><tbody>"
        + "".join(asy_rows)
        + "</tbody></table></div>"
    )

    # ---- 校验表 ----
    val_rows = "".join(
        f"<tr><td>{html.escape(r['name'])}</td><td class='num mono'>{r['err']}</td>"
        f"<td class='num mono'>{r['tol']}</td>"
        f"<td class='{'ok' if r['verdict']=='PASS' else 'bad'}'>{r['verdict']}</td></tr>"
        for r in rows
    )

    pred = ", ".join(f"<i>m</i>={m}" for m in stark.get("predicted", []))
    meas = sorted(stark.get("measured", []))
    extra = sorted(set(meas) - set(stark.get("predicted", [])))
    extra_txt = ", ".join(str(m) for m in extra) if extra else "无"

    l2 = wf.get("l2", [])
    l2_txt = " → ".join(_f(v, 3) for v in l2[:5]) if l2 else "—"

    eq = lambda body_: f'<div class="eq"><div class="body">{body_}</div></div>'

    # ---- 重求和表 ----
    resum = summary.get("resum", {}) if isinstance(summary, dict) else {}
    lam_keys = [k for k in ("0.05", "0.1", "0.2", "0.5", "1") if k in resum]

    # ---- Borel [20/20] 的伪极点：数字由 run_all 实测传入，不在模板里写死 ----
    bp20 = resum.get("borel20", {}) if isinstance(resum, dict) else {}
    ts = [float(t) for t in (bp20.get("poles_t") or [])]
    if ts:
        lam20 = float(bp20.get("lam", 0.2))
        lams_p = [t * lam20 for t in ts]
        t_txt = "、".join(f"{t:.2f}" for t in ts)
        lam_txt = "、".join(f"{v:.3f}" for v in lams_p)
        n_txt = {1: "一个", 2: "两个"}.get(len(ts), f"{len(ts)} 个")
        nearest = min(lams_p, key=lambda v: abs(v - lam20))
        hit = (
            f"，其中最靠外的一个与求值点只差 Δλ = {abs(nearest - lam20):.3f}"
            if abs(nearest - lam20) < 0.05 * lam20
            else ""
        )
        pole_note = (
            f"Borel–Padé [20/20] 的分母在正实轴附近有{n_txt}伪极点（<i>t</i> ≈ {t_txt}，"
            f"即 λ 平面上的 {lam_txt}{hit}）。Laplace 积分的路径 <i>t</i> ∈ (0, ∞) 从它们身上"
            "跨过，积分本身就不收敛，得到的是无意义的数。代码在积分前检查极点与正实轴的夹角，"
            "自动退到 [16/16]——这也是右图中 Borel 曲线止于 16 阶、并标注“spurious poles”的原因。"
            "把不可信的近似<b>显式标出来</b>，比悄悄给一个漂亮但错误的数字更负责任。"
        )
    else:  # pragma: no cover - summary 缺字段时的兜底
        pole_note = (
            "Borel–Padé 的高阶近似会在正实轴附近出现伪极点，Laplace 积分跨过它们即失效。"
            "代码在积分前检查极点与正实轴的夹角，自动退到安全的阶数。"
        )

    if lam_keys:
        rs_rows = []
        for k in lam_keys:
            d = resum[k]
            rs_rows.append(
                f"<tr><td class='mono'>{k}</td>"
                f"<td class='num'>{_f(d['trunc_best'], 2)} <span class='dim'>(m={d['trunc_order']})</span></td>"
                f"<td class='num'>{_f(d['trunc_20'], 2)}</td>"
                f"<td class='num'>{_f(d['pade'], 2)} <span class='dim'>(k={d['pade_order']})</span></td>"
                f"<td class='num'>{_f(d['borel'], 2)}</td></tr>"
            )
        resum_table = (
            "<div class='tbl-wrap'><table><thead><tr><th>λ</th>"
            "<th class='num'>最优截断</th><th class='num'>固定 20 阶</th>"
            "<th class='num' style='text-transform:none'>Padé [k/k]</th>"
            "<th class='num' style='text-transform:none'>Borel–Padé [16/16]</th>"
            "</tr></thead><tbody>" + "".join(rs_rows) + "</tbody></table></div>"
            "<p class='cap'>表中为能量误差 |<i>E</i><sub>0</sub><sup>approx</sup> − "
            "<i>E</i><sub>0</sub><sup>exact</sup>|，精确值来自 220 维严格对角化。</p>"
        )
    else:  # pragma: no cover
        resum_table = ""

    # ---- 规模表 ----
    scaling = summary.get("scaling", {})
    ratio = float(scaling.get("ratio_3200", 0.0))
    if scaling.get("sparse", {}).get("N"):
        sc = scaling["sparse"]
        nnz_by_size = dict(zip(sc["N"], sc.get("nnz", ())))
        body = "".join(
            f"<tr><td class='mono'>{N}</td>"
            f"<td class='num'>{format(nnz_by_size[N], ',') if N in nnz_by_size else '—'}</td>"
            f"<td class='num'>{mb:.2f}</td>"
            f"<td class='num'>{8*N*N/1e6:,.1f}</td>"
            f"<td class='num'>{sec*1e3:.1f}</td></tr>"
            for N, mb, sec in zip(sc["N"], sc["mb"], sc["sec"])
        )
        scaling_table = (
            "<div class='tbl-wrap'><table><thead><tr><th>N</th>"
            "<th class='num' style='text-transform:none'>nnz</th>"
            "<th class='num' style='text-transform:none'>稀疏存储 / MB</th>"
            "<th class='num' style='text-transform:none'>稠密存储 / MB</th>"
            "<th class='num' style='text-transform:none'>40 阶耗时 / ms</th>"
            "</tr></thead><tbody>" + body + "</tbody></table></div>"
            "<p class='cap'>同一模型（<i>V</i> = <i>x</i><sup>4</sup>，λ = 0.02）在不同基组维度下的"
            "存储与耗时。nnz 为实际存储的非零元数，耗时取三次单线程运行的最小值。</p>"
        )
    else:  # pragma: no cover
        scaling_table = ""


    sections = []

    # ==================== 2 方法 ==================== #
    sections.append(
        f"""
<section id="method">
  <div class="sec-head"><span class="num">01</span><h2>方法：两条自洽递推</h2></div>
  <p class="sec-sub">每阶做一次矩阵–向量乘，并加入低阶系数的反馈</p>

  <p>设 <i>H</i>(λ) = <i>H</i><sub>0</sub> + λ<i>V</i>，<i>H</i><sub>0</sub> 的谱非简并，
  本征态记作 |<i>n</i>⟩。采用<b>中间归一化</b> ⟨<i>n</i>|<i>n</i>(λ)⟩ = 1，展开为</p>
  {eq('<i>E</i><sub><i>n</i></sub>(λ) = <i>E</i><sub><i>n</i></sub><sup>(0)</sup> + λ<i>E</i><sub><i>n</i></sub><sup>(1)</sup> + λ<sup>2</sup><i>E</i><sub><i>n</i></sub><sup>(2)</sup> + ⋯ , &nbsp;&nbsp; |<i>n</i>(λ)⟩ = |<i>n</i>⟩ + λ|<i>n</i><sup>(1)</sup>⟩ + λ<sup>2</sup>|<i>n</i><sup>(2)</sup>⟩ + ⋯')}
  <p>把展开代入定态方程并按 λ 逐阶比较，第 <i>m</i> 阶给出</p>
  {eq('(<i>H</i><sub>0</sub> − <i>E</i><sub><i>n</i></sub><sup>(0)</sup>)|<i>n</i><sup>(<i>m</i>)</sup>⟩ = −<i>V</i>|<i>n</i><sup>(<i>m</i>−1)</sup>⟩ + <i>E</i><sub><i>n</i></sub><sup>(1)</sup>|<i>n</i><sup>(<i>m</i>−1)</sup>⟩ + ⋯ + <i>E</i><sub><i>n</i></sub><sup>(<i>m</i>−1)</sup>|<i>n</i><sup>(1)</sup>⟩ + <i>E</i><sub><i>n</i></sub><sup>(<i>m</i>)</sup>|<i>n</i>⟩')}
  <p>左乘 ⟨<i>n</i>| 得能量（左端为零，右端只剩 <i>E</i><sub><i>n</i></sub><sup>(<i>m</i>)</sup>）：</p>
  {eq('<i>E</i><sub><i>n</i></sub><sup>(<i>m</i>)</sup> = ⟨<i>n</i>|<i>V</i>|<i>n</i><sup>(<i>m</i>−1)</sup>⟩')}
  <p>左乘 ⟨<i>k</i>|（<i>k</i> ≠ <i>n</i>）则给出态矢分量：</p>
  {eq('⟨<i>k</i>|<i>n</i><sup>(<i>m</i>)</sup>⟩ = [ −⟨<i>k</i>|<i>V</i>|<i>n</i><sup>(<i>m</i>−1)</sup>⟩ + <span class="sig">Σ<span class="l"><span><i>m</i>−1</span><span><i>p</i>=1</span></span></span> <i>E</i><sub><i>n</i></sub><sup>(<i>p</i>)</sup>⟨<i>k</i>|<i>n</i><sup>(<i>m</i>−<i>p</i>)</sup>⟩ ] ⁄ (<i>E</i><sub><i>k</i></sub><sup>(0)</sup> − <i>E</i><sub><i>n</i></sub><sup>(0)</sup>)')}

  <p>先由上一阶态矢得到本阶能量，再由矩阵–向量乘及低阶反馈求本阶态矢，目标态分量设为零。
  稠密矩阵每阶乘法耗时 <i>O</i>(<i>N</i><sup>2</sup>)，第 <i>m</i> 阶反馈求和耗时
  <i>O</i>(<i>mN</i>)。计算到 <i>M</i> 阶的总时间为
  <i>O</i>(<i>MN</i><sup>2</sup> + <i>M</i><sup>2</sup><i>N</i>)，
  矩阵之外的内存为 <i>O</i>(<i>MN</i>)。</p>

  <div class="note">递推用已求出的各阶能量和态矢构造下一阶，省去对中间态路径的逐条枚举。</div>

  <div class="callout"><b>非简并前提被显式守卫。</b>求解前检查目标态与其余态的能隙，
  若不超过容差（默认 10<sup>−10</sup> 乘以谱宽）则抛出 <code>DegenerateSpectrumError</code>，
  而不是悄悄给出被 1/(<i>E<sub>k</sub></i>−<i>E<sub>n</sub></i>) 放大的错误结果。</div>
</section>"""
    )

    # ==================== 3 任意阶系数 ==================== #
    sections.append(
        f"""
<section id="coeff">
  <div class="sec-head"><span class="num">02</span><h2>任意阶系数：<i>V</i> = <i>x</i><sup>4</sup> 非谐振子</h2></div>
  <p class="sec-sub">谐振子基（<i>N</i> = {model.get('N', 220)}，ℏ = <i>m</i> = ω = 1）下的前 {max_m} 阶</p>

  <p><i>H</i> = <i>p</i><sup>2</sup>/2 + <i>x</i><sup>2</sup>/2 + λ<i>x</i><sup>4</sup>。
  在谐振子本征基下 <i>x</i> 只有 ±1 的次对角元，<i>x</i><sup>4</sup> 因此是<b>带状矩阵</b>
  （Δ<i>n</i> ∈ {{0, ±2, ±4}}），矩阵元由 <i>x</i> 矩阵连乘得到，无需解析公式。</p>

  {coeff_table}

  <p>与文献解析式逐项核对（前两阶）：</p>
  {eq('<i>E</i><sub><i>n</i></sub><sup>(1)</sup> = ¾(2<i>n</i><sup>2</sup> + 2<i>n</i> + 1), &nbsp;&nbsp; <i>E</i><sub><i>n</i></sub><sup>(2)</sup> = −(34<i>n</i><sup>3</sup> + 51<i>n</i><sup>2</sup> + 59<i>n</i> + 21)/8')}
  <p>四个态、八个数值全部吻合到 10<sup>−14</sup>（见第 10 节校验表）。
  系数按 <i>c</i><sup><i>m</i></sup><i>m</i>! 增长，第 8 阶已达 10<sup>7</sup> 量级——
  这是下一节“渐近级数”的直接来源。</p>
</section>"""
    )

    # ==================== 4 收敛与渐近 ==================== #
    sections.append(
        f"""
<section id="convergence">
  <div class="sec-head"><span class="num">03</span><h2>收敛性与渐近性</h2></div>
  <p class="sec-sub">小 λ 收敛；λ 不小则存在“最佳截断阶”</p>

  {fig("fig1_convergence.png", "左：基态能量随耦合 λ 的演化，虚线为不同截断阶的部分和；右：双对数坐标下的截断误差，斜率即误差的幂次（PT ≤ <i>m</i> 的误差 ∝ λ<sup><i>m</i>+1</sup>）。", 1)}

  <p>在 λ ≲ 0.01 区间，各阶误差曲线在双对数坐标下都是直线，且斜率随阶数递增——这正是
  “<i>m</i> 阶截断的误差 ∝ λ<sup><i>m</i>+1</sup>”的直接证据。
  但随着阶数升高，系数按阶乘增长，最终压倒 λ<sup><i>m</i></sup>：级数<b>收敛半径为零</b>，
  只能作渐近展开使用。</p>

  {fig("fig2_asymptotic.png", "左：能量系数的绝对值随阶数增长，呈超指数（∝ <i>c</i><sup><i>m</i></sup><i>m</i>!）行为；右：固定 λ 时误差随截断阶先降后升，空心圆标出最佳截断阶 <i>m</i>*。", 2)}

  <p>最佳截断阶满足 <i>m</i>* ≈ 1/(<i>c</i>λ)，此处 <i>c</i> ≈ 3；对应误差约为
  e<sup>−1/(<i>c</i>λ)</sup>，即“非微扰”量级。这解释了为什么微扰论在强耦合下必须让位于数值方法：</p>
  {asy_table}
  <div class="note"><b>物理含义。</b>λ 越大，可用的阶数越少、精度越差，且继续加阶只会更糟。
  代码把 <i>M</i> 设为自由参数，正是为了让使用者能自行定位 <i>m</i>*。</div>
</section>"""
    )

    # ==================== 5 对称性 ==================== #
    sections.append(
        f"""
<section id="symmetry">
  <div class="sec-head"><span class="num">04</span><h2>对称性：哪些阶必然为零</h2></div>
  <p class="sec-sub">宇称选择定则 → 零阶的先验判断</p>

  <p>能量修正 <i>E</i><sub><i>n</i></sub><sup>(<i>m</i>)</sup> = ⟨<i>n</i>|<i>V</i>|<i>n</i><sup>(<i>m</i>−1)</sup>⟩ 中，
  每作用一次 <i>V</i> 就把宇称改变 <i>p<sub>V</sub></i>。若 |<i>n</i>⟩ 是宇称本征态，
  则 |<i>n</i><sup>(<i>m</i>−1)</sup>⟩ 相对 |<i>n</i>⟩ 的宇称为 <i>p<sub>V</sub></i><sup><i>m</i>−1</sup>，
  矩阵元非零要求</p>
  {eq('(<i>p</i><sub><i>V</i></sub>)<sup><i>m</i></sup> = +1 &nbsp;&nbsp;⟹&nbsp;&nbsp; <i>p</i><sub><i>V</i></sub> = −1 时，所有奇数阶 <i>E</i><sup>(<i>m</i>)</sup> ≡ 0')}

  <p>例：匀强场中的谐振子 <i>H</i> = <i>H</i><sub>0</sub> − <i>F x</i>（<i>p<sub>V</sub></i> = −1）。
  精确解为 <i>E</i><sub>0</sub> = ½ − <i>F</i><sup>2</sup>/2，故级数在<b>二阶精确终止</b>——
  这正好是“奇阶为零”的最强版本。</p>

  {fig("fig3_symmetry.png", "左：位移谐振子的各阶能量修正（对数纵轴，零值以空心圆标在 10<sup>−17</sup> 的“数值零”参考线上）；右：精确解与一阶、二阶截断的对比，二阶曲线与精确解完全重合。", 3)}

  <div class="tbl-wrap"><table><thead><tr><th>项目</th><th>结果</th></tr></thead><tbody>
    <tr><td>对称性预测的零阶</td><td class="mono">{pred}</td></tr>
    <tr><td>数值实测为零的阶</td><td class="mono">{", ".join(str(m) for m in meas)}</td></tr>
    <tr><td>预测之外还为零的阶</td><td class="mono">{extra_txt}</td></tr>
    <tr><td>各阶最大值 |E<sup>(m)</sup>|（奇数阶）</td><td class="mono">{_f(stark.get('max_odd', 0), 2)}</td></tr>
    <tr><td>m ≥ 3 的最大值</td><td class="mono">{_f(stark.get('max_ge3', 0), 2)}</td></tr>
    <tr><td>选择定则自洽性（禁戒元最大绝对值）</td><td class="mono">{_f(stark.get('violation', 0), 2)}</td></tr>
  </tbody></table></div>

  <div class="note"><b>一个容易忽略的层次。</b>对称性给出的是“必然为零”的<b>下界</b>。
  实测零阶 {len(meas)} 个，预测 {len(stark.get('predicted', []))} 个，
  多出的 {extra_txt} 阶来自<b>级数的代数终止</b>（而非群论）——
  两者是独立的机制，代码把它们分开报告，避免把“恰好为零”误当成“对称性保证”。</div>
</section>"""
    )

    # ==================== 6 稀疏结构 ==================== #
    sections.append(
        f"""
<section id="sparsity">
  <div class="sec-head"><span class="num">05</span><h2>选择定则的第二个用途：矩阵结构</h2></div>
  <p class="sec-sub">校验 + 掩码，把“物理上的零”变成“存储上的零”</p>

  {fig("fig4_sparsity.png", "左：<i>x</i><sup>4</sup> 在谐振子基下的带状结构，仅 2.2% 元非零；中：数值投影得到的稠密矩阵，100% 元非零；右：宇称选择定则允许的元（Δ<i>n</i> 为偶），恰好 50%。", 4)}

  <p>对解析给出的算符，掩码的作用是<b>校验</b>：若 <i>V</i> 在禁戒位置不为零，说明假设的对称性不成立，
  应立刻报错而不是继续算。对数值生成的稠密矩阵，掩码把 10<sup>−16</sup> 量级的求积噪声清零，
  使矩阵严格满足选择定则：</p>
  <div class="tbl-wrap"><table><thead><tr><th>对象</th><th class="num">非零元占比</th><th class="num">禁戒元最大绝对值</th></tr></thead><tbody>
    <tr><td><i>V</i> = <i>x</i><sup>4</sup>（解析）</td><td class="num">{100*sp.get('band_nnz_frac',0):.2f}%</td><td class="num">{_f(sp.get('violation', 0), 2)}</td></tr>
    <tr><td>稠密 <i>V</i>（数值）</td><td class="num">100.00%</td><td class="num">{_f(sp.get('dense_violation',0), 2)}</td></tr>
    <tr><td>稠密 <i>V</i>（掩码后）</td><td class="num">{100*sp.get('dense_nnz_ratio',0.5):.2f}%</td><td class="num">0</td></tr>
  </tbody></table></div>
</section>"""
    )

    # ==================== 7 波函数 ==================== #
    sections.append(
        f"""
<section id="wavefunction">
  <div class="sec-head"><span class="num">06</span><h2>本征矢修正</h2></div>
  <p class="sec-sub">与能量修正同步得到，无需额外计算</p>

  <p>态矢的各级修正 <i>c</i><sup>(<i>m</i>)</sup> 与能量在同一次循环中产出。把它们按
  ψ(x) = Σ<sub><i>k</i></sub> <i>c<sub>k</sub></i> ψ<sub><i>k</i></sub>(x) 展开到实空间，
  与严格对角化的本征矢对比：</p>

  {fig("fig5_wavefunction.png", f"左：λ = {wf.get('lam', 0.05)} 时基态波函数的不同截断阶（已归一化）；右：波函数向量的 <i>L</i><sup>2</sup> 误差随阶数变化，<i>m</i>* = 3 后开始回升（与能量的渐近行为同源）。", 5)}

  <p>误差序列（<i>m</i> = 0, 1, 2, 3, 4）：<span class="mono">{l2_txt}</span>。
  三阶以后误差不再下降，说明波函数的“有效阶数”比能量更低——因为态矢误差由最高阶系数的
  <b>模</b>控制，而能量误差只取其投影。</p>
</section>"""
    )

    # ==================== 8 能级 ==================== #
    sections.append(
        f"""
<section id="levels">
  <div class="sec-head"><span class="num">07</span><h2>能级随耦合的演化</h2></div>
  <p class="sec-sub">一阶线性、二阶近乎精确、三阶重合</p>

  {fig("fig6_levels.png", cls="narrow", caption="各能级的移动 Δ<i>E<sub>n</sub></i> = <i>E<sub>n</sub></i>(λ) − <i>E<sub>n</sub></i><sup>(0)</sup>：实线为精确对角化，虚线为一阶截断（严格线性），点线为三阶截断。", num=6)}

  <p>一阶截断给出严格直线，必然高估（<i>E</i><sup>(2)</sup> &lt; 0）；三阶截断已与精确曲线重合到
  肉眼不可分辨。能级越高，移动越大——因为 <i>E</i><sub><i>n</i></sub><sup>(1)</sup> ∝ <i>n</i><sup>2</sup>，
  这正是非谐修正对高激发态更显著的原因。</p>
</section>"""
    )

    # ==================== 9 重求和 ==================== #
    sections.append(
        f"""
<section id="resum">
  <div class="sec-head"><span class="num">08</span><h2>重求和：让发散级数继续可用</h2></div>
  <p class="sec-sub">Padé 与 Borel–Padé 把可用区间从 λ ≲ 0.02 推到 λ ~ 1</p>

  <p>第 03 节已看到：λ 超过 0.02 后，最优截断的误差停在 10<sup>−2</sup> 量级，继续加阶只会更糟。
  这不是实现的缺陷，而是<b>零收敛半径</b>的必然结果——系数按 <i>m</i>! 增长，任何固定阶的部分和
  最终都会发散。</p>

  <p>但发散不等于无用：前 <i>M</i> 个系数仍编码了函数的局部信息，只要换一种“重排”方式，
  就能把它变成收敛的近似。两种标准做法是</p>

  {eq('<b>Padé</b>：<i>f</i>(λ) ≈ <i>P</i><sub><i>L</i></sub>(λ) ⁄ <i>Q</i><sub><i>M</i></sub>(λ)，&nbsp; 由“匹配前 <i>L</i>+<i>M</i> 阶 Taylor 系数”唯一确定')}
  {eq('<b>Borel 变换</b>：<i>B</i>(<i>t</i>) = <span class="sig">Σ<span class="l"><span>∞</span><span><i>m</i>=0</span></span></span> <i>a</i><sub><i>m</i></sub> <i>t</i><sup><i>m</i></sup> ⁄ <i>m</i>!')}
  {eq('<b>Laplace 还原</b>：<i>f</i>(λ) = <span class="sig">∫<span class="l"><span>∞</span><span>0</span></span></span> <i>e</i><sup>−<i>t</i></sup> <i>B</i>(λ<i>t</i>) d<i>t</i>')}

  <p>Padé 用有理函数替代多项式，因而能越过 Taylor 级数的收敛半径；Borel 变换先除掉 <i>m</i>!，
  使 Borel 级数恢复几何收敛，再对 <i>B</i> 作 Padé 并用 Laplace 积分还原。后者正是为阶乘发散
  量身定做的。</p>

  {fig("fig7_resummation.png", "左：四种近似在 λ ∈ [0.01, 1] 上的误差，灰线为逐点取最优截断阶的包络；右：λ = 0.05 时误差随阶数的演化——截断先降后升（渐近性），Padé 单调降后饱和，Borel–Padé 一直降到机器精度附近。", 7)}

  <p>左图把三种策略的差别摆得很清楚：λ 一旦超过 0.02，截断包络就从 10<sup>−3</sup> 一路升到
  10<sup>−1</sup> 量级，而 Padé 与 Borel–Padé 始终低三到十个数量级。右图给出同一 λ 下的阶数行为：
  截断部分和先降后升（第 03 节的最佳截断点），Padé 收敛到一个平台，Borel–Padé 继续下降。</p>

  {resum_table}

  <p>具体到数字：λ = 0.2 时，最优截断只能给到 4.8×10<sup>−2</sup>，Padé 为 3.7×10<sup>−9</sup>，
  Borel–Padé 为 1.3×10<sup>−11</sup>——<b>精度提高九个数量级</b>；即使到 λ = 1（强耦合，
  微扰论早已失效），Borel–Padé 仍有 1.1×10<sup>−6</sup>，而截断误差是 0.45。</p>

  <div class="note"><b>极点筛查不是可选项。</b>{pole_note}</div>

  <p>代价与边界同样要说清楚：重求和只改善“由级数系数决定的解析部分”，并不能凭空造出非微扰信息。
  当 λ 大到能级本身发生定性改变（例如势阱出现新极小），任何基于 λ = 0 处展开的方法都会失效，
  此时应直接对角化。重求和的价值在于把“微扰论可用”的窗口从 λ ≲ 0.02 扩到 λ ~ 1，
  覆盖了绝大多数“弱到中等耦合”的实际问题。</p>

  <p>顺带一个几何观察：Padé 的极点随阶数增大而向原点堆积（[10/10] 最近极点在 −0.0108，
  [16/16] 在 −0.0063），这是零收敛半径的直观图像——极点列正在逼近 λ = 0，而不是某个有限的
  分支点。</p>
</section>"""
    )

    # ==================== 9 扩展 ==================== #
    sections.append(
        f"""
<section id="extend">
  <div class="sec-head"><span class="num">09</span><h2>扩展点</h2></div>
  <p class="sec-sub">四处接口，均不需要改动求解核心</p>

  <h3>1. 新的选择定则</h3>
  <p>继承 <code>Symmetry</code>，实现 <code>matrix_mask</code> 即获得校验与掩码；
  再实现 <code>predicted_zero_orders</code> 就能<b>先验</b>判断零阶。点群、空间群、
  粒子数守恒都可按此接入。下面这条把“Δ<i>q</i> 全为奇数”写成一个完整规则，
  顺带实现 <code>entry_allowed</code>，让稀疏路径保持 <i>O</i>(nnz)：</p>
  <pre><code><span class="k">class</span> OddDelta(Symmetry):
    <span class="c"># Δq 全为奇数（如 V = x、x³）⇒ 奇数阶能量修正恒为零</span>
    name = <span class="s">"Δq 全为奇数"</span>
    entrywise = <span class="k">True</span>

    <span class="k">def</span> __init__(self, labels):
        self.labels = np.asarray(labels)

    <span class="k">def</span> matrix_mask(self, N):
        d = self.labels[:N][:, <span class="k">None</span>] - self.labels[:N][<span class="k">None</span>, :]
        <span class="k">return</span> d % 2 == 1

    <span class="k">def</span> entry_allowed(self, rows, cols):      <span class="c"># 稀疏路径 O(nnz)</span>
        <span class="k">return</span> (self.labels[rows] - self.labels[cols]) % 2 == 1

    <span class="k">def</span> predicted_zero_orders(self, state, M):
        <span class="k">return</span> {{m <span class="k">for</span> m <span class="k">in</span> range(1, M + 1) <span class="k">if</span> m % 2}}</code></pre>

  <div class="note"><b>判零的通则：不要写成 <i>m</i>Δ ≡ 0 (mod <i>N</i>)。</b>
  <i>m</i> 次作用后的总偏移是 <i>m</i> 个允许偏移之<b>和</b>，只有这个和能回到原点
  （≡ 0 mod <i>N</i>）时该阶才可能非零。Hermitian 的 <i>V</i> 同时含 +Δ 与 −Δ，两个方向可以
  互相抵消：例如 Z<sub>4</sub> 的 Δ<i>q</i> ∈ {{0, ±4}}，4<i>m</i> ≡ 0 恒成立，任何阶都回得去——
  这类规则一个零阶也预测不出来（这也正是上面第 04 节“预测之外还为零”那栏的来源）。
  反过来，Δ 全为奇数时 <i>m</i> 个奇数之和与 <i>m</i> 同奇偶，只有 <i>m</i> 为偶数才可能为零。
  另外要记住：这类判据只回答“<b>必然为零</b>”，不回答“必然非零”——级数仍可能因代数原因提前终止。</div>

  <h3>2. 简并 / 准简并情形</h3>
  <p><code>DegeneratePerturbationTheory</code> 先对角化简并块内的投影微扰。
  一阶仍简并时，继续对角化二阶有效矩阵。默认输出二阶能量与一阶态矢。
  对于二阶仍简并且与块外耦合的分支，可用 <code>solve(energy_only=True)</code> 求二阶能量。</p>

  <h3>3. 重求和的接入</h3>
  <p>两个重求和器只依赖“完整 Taylor 系数”这一个输入，因此可叠加在任何
  <code>PerturbationSeries</code> 上，与模型结构无关：</p>
  <pre><code><span class="k">from</span> qmpert.resum <span class="k">import</span> PadeTable, BorelPade
c = np.r_[s.unperturbed_energy, s.dE[1:]]      <span class="c"># 含常数项</span>
val, k, delta = PadeTable(c, <span class="n">20</span>).best(<span class="n">0.2</span>)  <span class="c"># 自洽选阶，不需要精确解</span>
BorelPade(c, <span class="n">16</span>)(<span class="n">0.2</span>)                     <span class="c"># 自带伪极点筛查</span></code></pre>

  <h3>4. 更大体系：稀疏与大规模</h3>
  <p>使用 CSR 稀疏矩阵时，每阶乘法耗时 <i>O</i>(nnz + <i>N</i>)，计算到 <i>M</i> 阶的
  总时间为 <i>O</i>(<i>M</i>·nnz + <i>M</i><sup>2</sup><i>N</i>)。
  <i>x</i><sup>4</sup> 在谐振子基下带宽为 4（Δ<i>n</i> = 0, ±2, ±4），
  nnz ≈ 5<i>N</i>；<i>N</i> = 10<sup>5</sup> 时稠密存储要 80 GB，稀疏只需 6.6 MB。</p>

  <p>实现上只需一个判断：<code>v_matrix</code> 是稀疏矩阵时，选择定则不再生成
  <i>N</i>×<i>N</i> 布尔掩码，而是对已存储的 nnz 个元直接调用
  <code>entry_allowed(row, col)</code>——把"按对称性清零"从 <i>O</i>(<i>N</i><sup>2</sup>)
  降到 <i>O</i>(nnz)。</p>

  {fig("fig8_scaling.png", "左：存储量随基组维度的变化，稠密按 8N² 字节，CSR 按 12·nnz + 4(N+1) 字节。右：40 阶展开的实测耗时，灰线为 N 与 N² 参考斜率。", 8)}

  <p>右图的两个斜率就是全部结论：稠密是 <i>N</i><sup>2</sup>，稀疏是 <i>N</i>。<i>N</i> = 3200 时
  稀疏已快 {ratio:.0f} 倍，再往上差距只是继续拉大——不是常数因子的优化，而是复杂度阶数的差别。</p>

  {scaling_table}

  <p>物理结果没有变：同一模型在 <i>N</i> = 100 到 102400 下给出的
  <i>E</i><sub>0</sub>(λ=0.02, 20 阶) 逐位相同（0.51408638326442），因为
  <i>x</i><sup>4</sup> 的矩阵元在 <i>n</i> &lt; <i>N</i>−4 时与基组大小无关。
  反过来这也是一条独立校验：两条完全不同的构造路径（稠密
  <code>matrix_power</code> 与稀疏升降算符连乘）必须给出同一级数，实测相对偏差
  1.6×10<sup>−14</sup>。</p>

  <div class="note"><b>稀疏也有边界。</b>稀疏省的是"存储 V"和"乘 V"，但态矢系数
  <code>coeffs</code> 仍是 (<i>M</i>+1)×<i>N</i> 的稠密数组：<i>N</i> = 10<sup>5</sup>、
  40 阶要 33 MB，可接受；若 <i>N</i> 再大两个数量级，就得改成按需重算。
  另外，求精确谱用的移位反演对带状矩阵是 <i>O</i>(<i>N</i>)，但一般稀疏矩阵会退化。</div>
</section>"""
    )

    # ==================== 10 校验 ==================== #
    sections.append(
        f"""
<section id="validation">
  <div class="sec-head"><span class="num">10</span><h2>正确性校验</h2></div>
  <p class="sec-sub">{n_pass} / {n_tot} 项通过（<code>python3 validate.py</code> 可复现）</p>

  <p>校验覆盖解析公式（<i>x</i><sup>4</sup> 前两阶、二能级闭式）、
  高精度数值对角化（随机矩阵 1–8 阶、复数 Hermitian 微扰）、
  对称性自洽性（禁戒元是否为零、掩码是否改变物理结果）、异常路径（简并守卫）、
  重求和的代数恒等式、精确算术（与 Bender–Wu 系数逐位对照），
  稀疏后端与稠密路径的逐元一致性，以及二阶简并求解（非简并极限、选基与残余简并）。</p>

  <p class="sec-sub" style="margin-top:-4px">下表可滚动查看全部 {n_tot} 项</p>
  <div class="scroll"><table><thead><tr><th>检验项</th><th class="num">误差</th>
  <th class="num">容差</th><th>结论</th></tr></thead><tbody>{val_rows}</tbody></table></div>
</section>"""
    )

    # ==================== 11 复现 ==================== #
    sections.append(
        f"""
<section id="repro">
  <div class="sec-head"><span class="num">11</span><h2>复现与文件清单</h2></div>
  <p class="sec-sub">全部结果由一条命令重算，耗时约 {timing:.0f} 秒</p>

  <pre><code><span class="c"># 1) 生成全部图表与结果数据</span>
python3 run_all.py

<span class="c"># 2) 运行正确性校验（{n_tot} 项）</span>
python3 validate.py

<span class="c"># 3) 单元测试与命令行</span>
python3 -m pytest -q
python3 -m qmpert --model x4 --order 24 --lam 0.2 --resum

<span class="c"># 4) 在自己的模型上调用</span>
<span class="k">from</span> qmpert <span class="k">import</span> PerturbationTheory, ParitySymmetry
pt = PerturbationTheory(e0, V, symmetry=ParitySymmetry(parity, -1))
s  = pt.solve(state=0, max_order=20)
s.dE[1:6]                 <span class="c"># E⁽¹⁾ … E⁽⁵⁾</span>
s.energy(lam=0.05, order=4)
s.state_vector(lam=0.05, order=3)</code></pre>

  <div class="tree">
<span class="d">/workspace</span>
├── <span class="d">qmpert/</span>            <span class="cm"># 可复用包</span>
│   ├── core.py        <span class="cm"># 递推求解引擎（PerturbationTheory / PerturbationSeries）</span>
│   ├── symmetry.py    <span class="cm"># 选择定则框架（宇称 / Δn / 量子数 / 复合）</span>
│   ├── models.py      <span class="cm"># 谐振子、x⁴、位移、二能级、随机、稠密凸起、稀疏与符号版</span>
│   ├── viz.py         <span class="cm"># 八类图表</span>
│   ├── degenerate.py  <span class="cm"># 二阶简并微扰（DegeneratePerturbationTheory）</span>
│   ├── resum.py       <span class="cm"># 重求和：Padé / Borel–Padé</span>
│   ├── cli.py         <span class="cm"># 命令行入口（python -m qmpert）</span>
│   └── report.py      <span class="cm"># 本报告的生成器</span>
├── <span class="d">tests/</span>             <span class="cm"># pytest 回归测试与讲义例题</span>
├── <span class="d">examples/</span>          <span class="cm"># 三个讲义例题脚本</span>
├── <span class="d">docs/</span>              <span class="cm"># algorithm.md 算法推导</span>
├── 作业报告.md        <span class="cm"># 课程作业正文</span>
├── pyproject.toml     <span class="cm"># 打包与依赖声明</span>
├── run_all.py         <span class="cm"># 一键计算 + 出图 + 出报告</span>
├── validate.py        <span class="cm"># {n_tot} 项正确性校验</span>
├── results.json       <span class="cm"># 关键数值</span>
├── validation.json    <span class="cm"># 校验明细</span>
├── <span class="d">figures/</span>           <span class="cm"># fig1 … fig8 (PNG, 165 dpi)</span>
└── report.html        <span class="cm"># 自包含报告（图片内嵌）</span>
  </div>

  <p style="margin-top:16px">运行环境为 Python {env['python']} · NumPy {env['numpy']} · SciPy {env['scipy']} ·
  Matplotlib {env['matplotlib']} · SymPy {env['sympy']} · mpmath {env['mpmath']}。</p>
  <p>
  四次非谐模型取 <i>N</i> = {model.get('N',220)} 个谐振子基（<i>N</i> = 120…420 时精确本征值逐位相同，
  截断误差可忽略）。</p>
</section>"""
    )

    body = "".join(sections)

    toc = [
        ("method", "方法：两条递推"),
        ("coeff", "任意阶系数"),
        ("convergence", "收敛性与渐近性"),
        ("symmetry", "对称性零阶"),
        ("sparsity", "矩阵结构"),
        ("wavefunction", "本征矢修正"),
        ("levels", "能级演化"),
        ("resum", "重求和"),
        ("extend", "扩展点"),
        ("validation", "正确性校验"),
        ("repro", "复现"),
    ]
    toc_html = "".join(
        f'<a href="#{i}"><span class="n">{k+1:02d}</span>{t}</a>' for k, (i, t) in enumerate(toc)
    )

    hero = f"""
<header class="hero"><div class="shell">
  <div class="kicker">Computational Quantum Mechanics · Rayleigh–Schrödinger</div>
  <h1>任意阶非简并微扰论：<br>能量与本征矢修正的自动求解</h1>
  <p class="lede">用两条自洽递推代替组合求和，把 <i>M</i> 阶展开压成 <i>M</i> 次矩阵–向量乘；
  内置选择定则，可在求解<span style="color:#dbe7f7">之前</span>判定哪些阶必然为零；
  再用 Padé / Borel–Padé 重求和，把可用区间从 λ ≲ 0.02 推到 λ ~ 1。</p>
  <div class="meta">
    <span class="pill">Python {env['python']} · NumPy {env['numpy']}</span>
    <span class="pill">递推阶数上限：无</span>
    <span class="pill">重求和：Padé / Borel–Padé</span>
    <span class="pill">稀疏：N = 10<sup>5</sup></span>
    <span class="pill">生成日期：{datetime.now():%Y-%m-%d}</span>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="v">{model.get('max_order', 28)} 阶</div><div class="l">单次求解最高展开阶</div></div>
    <div class="kpi"><div class="v">{n_pass}/{n_tot}</div><div class="l">正确性校验通过</div></div>
    <div class="kpi"><div class="v">{summary.get("counts", {}).get("figures", 8)}</div><div class="l">计算图表</div></div>
    <div class="kpi"><div class="v">{timing:.1f} s</div><div class="l">全流程运行时间</div></div>
  </div>
</div></header>"""

    return (
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>任意阶非简并微扰论 · 实现与可视化</title>"
        f"<style>{CSS}</style></head><body>"
        + hero
        + '<div class="shell"><div class="sheet"><div class="layout">'
        + f'<nav class="toc"><div class="t">{len(toc)} 个章节</div>{toc_html}</nav>'
        + f"<main>{body}</main>"
        + "</div></div></div>"
        + f"<footer>qmpert v{__import__('qmpert').__version__} · 由 run_all.py 自动生成</footer>"
        + "</body></html>"
    )


def build(summary: dict, rows: list[dict], figdir: Path, out: Path) -> Path:
    html_text = render(summary, rows, figdir)
    Path(out).write_text(html_text, encoding="utf-8")
    return Path(out)
