from __future__ import annotations

import re

from qmpert.cli import main


def test_exact_two_level_prints_fractions(capsys):
    assert main([
        "--model", "two-level", "--exact", "--order", "2", "--lam", "0.5",
    ]) == 0
    out = capsys.readouterr().out

    assert "-1/2" in out
    assert "-1/4" in out
    assert "-9/16" in out
    energy_rows = [line for line in out.splitlines() if re.match(r"^\s*\d+\s+", line)]
    assert energy_rows
    assert all(not re.search(r"\d(?:\.\d+)?e[+-]\d+", line, re.I) for line in energy_rows)


def test_show_state_prints_order_normalized_and_truncated_states(capsys):
    assert main([
        "--model", "two-level", "--show-state", "--order", "3", "--lam", "0.5",
    ]) == 0
    out = capsys.readouterr().out

    assert "逐阶范数归一化修正" in out
    assert "m=0" in out
    assert "m=1" in out
    assert "m=2" in out
    assert "  m=3\n" not in out
    assert "截断并整体归一化的态矢" in out


def test_plot_creates_nonempty_png(tmp_path, capsys):
    path = tmp_path / "series.png"
    assert main([
        "--model", "two-level", "--order", "3", "--lam", "0.2",
        "--plot", str(path),
    ]) == 0
    capsys.readouterr()

    assert path.is_file()
    assert path.stat().st_size > 0
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
