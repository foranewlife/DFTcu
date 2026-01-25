"""
DFTcu 配置基类

提供 YAML 加载、验证和单位转换功能。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from .grid import GridConfig
    from .physics import PhysicsConfig
    from .solver import SolverConfig
    from .task import TaskConfig


class DFTcuBaseConfig(BaseModel):
    """DFTcu 配置基类

    特性:
        - 支持 YAML 加载
        - 自动类型验证
        - 单位转换
        - 默认值管理
    """

    model_config = ConfigDict(
        extra="forbid",  # 禁止额外字段
        validate_assignment=True,  # 赋值时验证
        use_enum_values=True,  # 枚举使用值
    )

    @classmethod
    def from_yaml(cls, yaml_file: str | Path) -> "DFTcuBaseConfig":
        """从 YAML 文件加载配置

        Args:
            yaml_file: YAML 配置文件路径

        Returns:
            配置对象实例

        Raises:
            FileNotFoundError: 文件不存在
            yaml.YAMLError: YAML 格式错误
            pydantic.ValidationError: 配置验证失败
        """
        yaml_path = Path(yaml_file)
        if not yaml_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {yaml_file}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    def to_yaml(self, yaml_file: str | Path) -> None:
        """导出配置到 YAML 文件

        Args:
            yaml_file: 输出文件路径
        """
        yaml_path = Path(yaml_file)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(mode="json"),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.model_dump(mode="json")


class DFTcuConfig(DFTcuBaseConfig):
    """DFTcu 顶层配置

    包含所有子配置模块。
    """

    # 结构文件
    structure_file: str = Field(description="结构文件路径（VASP POSCAR 格式）")

    # 赝势定义 (顶级，简化格式)
    # 格式: { "Si": "Si.pz-rrkj.UPF", "O": "O.UPF" }
    pseudopotentials: Dict[str, str] = Field(description="元素到赝势文件名的映射")

    pseudo_dir: str = Field(default="./", description="赝势文件目录")

    # 必需配置
    grid: "GridConfig"
    physics: "PhysicsConfig"
    task: "TaskConfig"

    # 可选配置
    solver: Optional["SolverConfig"] = None

    # 元数据
    version: str = "1.0.0"
    description: Optional[str] = None

    # ═══════════════════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════════════════

    def get_pseudo_path(self, element: str) -> Path:
        """获取元素的赝势完整路径"""
        if element not in self.pseudopotentials:
            raise ValueError(f"元素 {element} 没有定义赝势")
        return Path(self.pseudo_dir) / self.pseudopotentials[element]

    def get_mass(self, element: str) -> float:
        """获取原子质量（amu），目前从 ASE 自动推断"""
        from ase.data import atomic_masses, chemical_symbols

        try:
            # 规范化元素符号
            sym = element.capitalize()
            idx = chemical_symbols.index(sym)
            return float(atomic_masses[idx])
        except (ValueError, IndexError):
            raise ValueError(f"无法推断元素 {element} 的质量")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """验证版本号格式"""
        parts = v.split(".")
        if len(parts) != 3:
            raise ValueError(f"版本号格式错误: {v}，应为 X.Y.Z")
        if not all(p.isdigit() for p in parts):
            raise ValueError(f"版本号必须为数字: {v}")
        return v
