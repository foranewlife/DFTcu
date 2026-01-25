"""
DFTcu Configuration Models

提供基于 Pydantic 的配置验证和 YAML 解析功能。
"""

# 必须在导入子模块后再导入 DFTcuConfig，以解析 forward references
from .base import DFTcuConfig
from .grid import GridConfig
from .physics import OccupationType, PhysicsConfig, SmearingType
from .solver import DavidsonConfig, MixingConfig, SolverConfig
from .task import NSCFConfig, SCFConfig, TaskConfig, TaskType

# 重建模型以解析 forward references
DFTcuConfig.model_rebuild()

__all__ = [
    "DFTcuConfig",
    "GridConfig",
    "PhysicsConfig",
    "OccupationType",
    "SmearingType",
    "TaskConfig",
    "TaskType",
    "SCFConfig",
    "NSCFConfig",
    "SolverConfig",
    "DavidsonConfig",
    "MixingConfig",
]
