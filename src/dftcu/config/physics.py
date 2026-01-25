"""
Physics 配置模型

定义系统的物理参数（电子数、能带、泛函、占据方式）。
"""

from enum import Enum
from typing import List, Optional

from pydantic import Field

from .base import DFTcuBaseConfig


class OccupationType(str, Enum):
    """占据数类型"""

    FIXED = "fixed"  # 固定占据（绝缘体）
    SMEARING = "smearing"  # Smearing（金属）


class SmearingType(str, Enum):
    """Smearing 类型"""

    GAUSSIAN = "gaussian"
    FERMI_DIRAC = "fermi_dirac"
    METHFESSEL_PAXTON = "methfessel_paxton"


class PhysicsConfig(DFTcuBaseConfig):
    """Physics 配置

    定义物理系统参数。

    Examples:
        >>> physics = PhysicsConfig(
        ...     nelec=8.0,
        ...     nbands=4,
        ...     xc_functional="lda",
        ...     occupations="fixed"
        ... )
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # 电子与能带
    # ═══════════════════════════════════════════════════════════════════════════

    nelec: float = Field(gt=0, description="电子数")

    nbands: int = Field(gt=0, description="能带数")

    # ═══════════════════════════════════════════════════════════════════════════
    # 物理模型
    # ═══════════════════════════════════════════════════════════════════════════

    xc_functional: str = Field(default="lda", description="交换相关泛函（'lda', 'pbe' 等）")

    occupations: OccupationType = Field(default=OccupationType.FIXED, description="占据数类型")

    smearing: Optional[SmearingType] = Field(default=None, description="Smearing 类型")

    degauss: Optional[float] = Field(default=None, gt=0, description="Smearing 宽度 (eV)")

    # ═══════════════════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════════════════

    def get_ase_atoms(self, structure_file: str):
        """从结构文件读取 ASE Atoms 对象"""
        from ase.io import read

        return read(structure_file)

    def get_atom_types(self, structure_file: str, pseudopotentials: dict) -> List[int]:
        """获取结构中每个原子的类型索引"""
        atoms = self.get_ase_atoms(structure_file)
        symbols = atoms.get_chemical_symbols()
        unique_elements = list(pseudopotentials.keys())

        types = []
        for sym in symbols:
            if sym not in unique_elements:
                raise ValueError(f"元素 {sym} 没有对应的赝势定义")
            types.append(unique_elements.index(sym))
        return types
