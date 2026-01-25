"""
Grid 配置模型

定义 FFT 网格、截断能。晶格信息从 POSCAR 文件读取。
"""

from typing import List, Optional

from pydantic import Field, field_validator, model_validator

from .base import DFTcuBaseConfig


class GridConfig(DFTcuBaseConfig):
    """Grid 配置

    定义 FFT 网格和截断能。晶格信息从 POSCAR 文件读取。

    Examples:
        >>> grid = GridConfig(
        ...     nr=[15, 15, 15],
        ...     ecutwfc=163.0,  # eV
        ...     ecutrho=652.0,  # eV
        ...     is_gamma=True
        ... )
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # FFT 网格
    # ═══════════════════════════════════════════════════════════════════════════

    nr: List[int] = Field(
        description="FFT 网格点数 [nr1, nr2, nr3]",
        min_length=3,
        max_length=3,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 截断能（单位：eV，用户友好）
    # ═══════════════════════════════════════════════════════════════════════════

    ecutwfc: float = Field(gt=0, description="波函数截断能 (eV)，内部转换为 Hartree")

    ecutrho: Optional[float] = Field(
        default=None,
        gt=0,
        description="密度截断能 (eV)，默认 = 4 × ecutwfc",
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 其他选项
    # ═══════════════════════════════════════════════════════════════════════════

    is_gamma: bool = Field(default=True, description="是否为 Gamma-only 计算")

    # ═══════════════════════════════════════════════════════════════════════════
    # 验证器
    # ═══════════════════════════════════════════════════════════════════════════

    @field_validator("nr")
    @classmethod
    def validate_nr(cls, v: List[int]) -> List[int]:
        """验证 FFT 网格"""
        if any(n <= 0 for n in v):
            raise ValueError(f"FFT 网格必须为正整数: {v}")
        return v

    @model_validator(mode="after")
    def set_default_ecutrho(self) -> "GridConfig":
        """设置默认 ecutrho = 4 × ecutwfc"""
        if self.ecutrho is None:
            self.ecutrho = 4.0 * self.ecutwfc
        return self

    # ═══════════════════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════════════════

    def get_lattice_bohr(self, structure_file: str):
        """从结构文件读取晶格向量（Bohr 单位）"""
        from ase.io import read

        from dftcu.constants import ANGSTROM_TO_BOHR

        atoms = read(structure_file)
        return atoms.get_cell()[:] * ANGSTROM_TO_BOHR

    def get_ecutwfc_hartree(self) -> float:
        """获取波函数截断能（Hartree 单位）"""
        from dftcu.constants import EV_TO_HA

        return self.ecutwfc * EV_TO_HA

    def get_ecutrho_hartree(self) -> float:
        """获取密度截断能（Hartree 单位）"""
        from dftcu.constants import EV_TO_HA

        return self.ecutrho * EV_TO_HA
