"""
Task 配置模型

定义计算任务（SCF, NSCF）和执行参数。
"""

from enum import Enum
from typing import Optional

from pydantic import Field, field_validator, model_validator

from .base import DFTcuBaseConfig


class TaskType(str, Enum):
    """任务类型"""

    SCF = "scf"  # 自洽场计算
    NSCF = "nscf"  # 非自洽场计算
    RELAX = "relax"  # 结构优化
    BANDS = "bands"  # 能带计算


class SCFConfig(DFTcuBaseConfig):
    """SCF 任务细节"""

    max_iterations: int = Field(default=100, gt=0)
    conv_thr: float = Field(default=1e-6, gt=0)


class NSCFConfig(DFTcuBaseConfig):
    """NSCF 任务细节"""

    charge_density_file: Optional[str] = Field(default=None)
    diagonalization: str = Field(default="davidson")
    diago_thr: float = Field(default=1e-10, gt=0)


class TaskConfig(DFTcuBaseConfig):
    """Task 配置

    定义任务类型和执行环境。
    """

    type: TaskType = Field(description="任务类型")

    # 任务细节
    scf: Optional[SCFConfig] = Field(default=None)
    nscf: Optional[NSCFConfig] = Field(default=None)

    # I/O
    outdir: str = Field(default="./output", description="输出目录")
    verbosity: str = Field(default="high", description="输出详细程度")

    @model_validator(mode="after")
    def validate_task_config(self) -> "TaskConfig":
        if self.type == TaskType.SCF and self.scf is None:
            self.scf = SCFConfig()
        if self.type == TaskType.NSCF and self.nscf is None:
            # NSCF 细节暂时设为默认
            if self.nscf is None:
                self.nscf = NSCFConfig()
        return self

    @field_validator("verbosity")
    @classmethod
    def validate_verbosity(cls, v: str) -> str:
        allowed = ["low", "medium", "high"]
        if v not in allowed:
            raise ValueError(f"verbosity 必须为 {allowed} 之一")
        return v
