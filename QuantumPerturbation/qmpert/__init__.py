"""
qmpert —— 非简并与二阶简并微扰论（Rayleigh–Schrödinger）工具箱
=============================================================

快速上手
--------
>>> import numpy as np
>>> from qmpert import PerturbationTheory, anharmonic_oscillator
>>> m = anharmonic_oscillator(N=200)
>>> pt = PerturbationTheory(m.e0, m.V, name=m.name)
>>> s = pt.solve(state=0, max_order=10)
>>> s.dE[1], s.dE[2]          # E₀⁽¹⁾ = 3/4, E₀⁽²⁾ = −21/8
(0.75, -2.625)
>>> s.energy(lam=0.05, order=4)         # 截断到 4 阶的能量
>>> s.state_vector(lam=0.05, order=3)   # 截断到 3 阶的态矢（未微扰基）

模块
----
core       PerturbationTheory / PerturbationSeries：非简并递推与范数归一化
degenerate DegeneratePerturbationTheory：简并块好基、二阶能量与一阶态矢
symmetry   选择定则框架（宇称、Δn、一般量子数、复合），可预测零阶
models     谐振子、非谐 x⁴、位移谐振子（Stark）、二能级、随机矩阵
viz        matplotlib 绘图（收敛性、渐近性、对称性、波函数、能级、重求和、规模）
resum      Padé / Borel–Padé 重求和
cli        命令行：python -m qmpert --help
report     生成自包含 HTML 报告
"""

from .core import DegenerateSpectrumError, PerturbationSeries, PerturbationTheory
from .degenerate import (
    DegenerateBranch,
    DegeneratePerturbationResult,
    DegeneratePerturbationTheory,
    ResidualDegeneracyError,
)
from .models import (
    Model,
    anharmonic_oscillator,
    banded_anharmonic,
    displaced_oscillator,
    gaussian_bump,
    harmonic_oscillator,
    ho_basis_matrix,
    ho_wavefunction,
    random_model,
    symbolic_anharmonic,
    two_level,
)
from .symmetry import (
    CompositeSymmetry,
    DeltaSymmetry,
    NoSymmetry,
    ParitySymmetry,
    QuantumNumberSymmetry,
    Symmetry,
)

__version__ = "1.2.0"

__all__ = [
    "PerturbationTheory",
    "PerturbationSeries",
    "DegenerateSpectrumError",
    "DegeneratePerturbationTheory",
    "DegeneratePerturbationResult",
    "DegenerateBranch",
    "ResidualDegeneracyError",
    "Symmetry",
    "NoSymmetry",
    "ParitySymmetry",
    "DeltaSymmetry",
    "QuantumNumberSymmetry",
    "CompositeSymmetry",
    "Model",
    "harmonic_oscillator",
    "anharmonic_oscillator",
    "displaced_oscillator",
    "two_level",
    "random_model",
    "banded_anharmonic",
    "symbolic_anharmonic",
    "ho_wavefunction",
    "ho_basis_matrix",
    "gaussian_bump",
]
