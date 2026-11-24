"""报告第 05 节的选择定则数据来源回归测试。"""

from pathlib import Path

import numpy as np
import pytest

from qmpert import DeltaSymmetry, ParitySymmetry, report


def test_analytic_quartic_row_uses_its_own_selection_rule():
    quartic = np.array([[1.0, 0.25], [0.25, 1.0]])
    stark = np.eye(2) * 0.75
    summary = {
        "sparsity": {
            "band_nnz_frac": 1.0,
            "violation": DeltaSymmetry({0, 2, -2, 4, -4}).validate(quartic),
        },
        "stark": {"violation": ParitySymmetry([1, -1], -1).validate(stark)},
    }
    html = report.render(summary, [], Path(__file__).parent / "unused_figures")
    quartic_row = html.split("<sup>4</sup>（解析）</td>", 1)[1].split("</tr>", 1)[0]

    assert '<td class="num">0.25</td>' in quartic_row


@pytest.mark.parametrize("recorded_nnz, expected", [(482, "482"), (None, "—")])
def test_scaling_table_renders_measured_nonzeros(recorded_nnz, expected):
    sparse = {"N": [100], "mb": [0.006], "sec": [0.001]}
    if recorded_nnz is not None:
        sparse["nnz"] = [recorded_nnz]
    html = report.render({"scaling": {"sparse": sparse}}, [], Path(__file__).parent)

    assert f"<td class='mono'>100</td><td class='num'>{expected}</td>" in html
