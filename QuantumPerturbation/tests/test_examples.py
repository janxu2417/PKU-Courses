"""课程例题的端到端回归：精确输出与直接出图。"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
EXAMPLES = PROJECT / "examples"


def run_example(name: str, *args: object) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT)
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name), *(str(a) for a in args)],
        cwd=PROJECT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )


def test_two_level_example_prints_exact_coefficients_and_plots(tmp_path):
    output = tmp_path / "two_level.png"
    result = run_example("ex1_two_level.py", "--output", output)
    assert result.returncode == 0, result.stderr
    assert "-1/4" in result.stdout
    assert "-1/8" in result.stdout
    assert output.exists() and output.stat().st_size > 0


def test_three_level_example_matches_lecture_and_plots(tmp_path):
    output = tmp_path / "three_level.png"
    result = run_example("ex3_three_level_degenerate.py", "--output", output)
    assert result.returncode == 0, result.stderr
    assert "(0, -10)" in result.stdout
    assert "(-2, 1)" in result.stdout
    assert "(2, 9)" in result.stdout
    assert "sqrt(2)" in result.stdout
    assert output.exists() and output.stat().st_size > 0


def test_delta_well_example_prints_distinct_errors_and_plots(tmp_path):
    output = tmp_path / "delta_well.png"
    result = run_example("ex2_delta_well.py", "--output", output, "--basis", 500)

    assert result.returncode == 0, result.stderr
    assert "基组误差" in result.stdout
    assert "阶数误差" in result.stdout
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_wavefunction_comparison_ignores_eigenvector_phase(tmp_path):
    from qmpert import PerturbationTheory, anharmonic_oscillator
    from qmpert.viz import fig_wavefunction

    model = anharmonic_oscillator(N=8)
    series = PerturbationTheory(model.e0, model.V).solve(0, 0)
    # 负号是同一归一化本征态的另一相位约定。
    model.exact_state = lambda n, lam: -np.eye(8)[n]
    errors = fig_wavefunction(model, series, 0, 0, np.linspace(-4, 4, 50), (0,),
                              tmp_path / "wavefunction.png")
    assert errors == [0.0]
