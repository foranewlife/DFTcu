"""
Solver 配置模型

定义求解器参数。
"""

from typing import Optional

from pydantic import Field, field_validator

from .base import DFTcuBaseConfig


class DavidsonConfig(DFTcuBaseConfig):
    """Davidson 对角化配置

    Examples:
        >>> davidson = DavidsonConfig(
        ...     max_iterations=100,
        ...     conv_thr=1e-10,
        ...     notconv_thr=2
        ... )
    """

    max_iterations: int = Field(default=100, gt=0, description="最大迭代次数")

    conv_thr: float = Field(default=1e-10, gt=0, description="对角化收敛阈值 (eV)，内部转换为 Hartree")

    notconv_thr: int = Field(default=2, ge=0, description="允许未收敛的能带数")

    diago_david_ndim: int = Field(default=4, gt=1, description="Davidson 子空间维度倍数")


class MixingConfig(DFTcuBaseConfig):
    """密度混合配置

    Examples:
        >>> mixing = MixingConfig(
        ...     mode="broyden",
        ...     beta=0.7,
        ...     ndim=8
        ... )
    """

    mode: str = Field(default="plain", description="混合模式（'plain', 'broyden'）")

    beta: float = Field(default=0.7, gt=0, le=1.0, description="混合参数")

    ndim: int = Field(default=8, gt=0, description="Broyden 混合历史维度")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        """验证混合模式"""
        allowed = ["plain", "broyden"]
        if v not in allowed:
            raise ValueError(f"mode 必须为 {allowed} 之一，当前: {v}")
        return v


class SolverConfig(DFTcuBaseConfig):
    """Solver 配置

    定义求解器参数。

    Examples:
        >>> solver = SolverConfig(
        ...     davidson=DavidsonConfig(max_iterations=100),
        ...     mixing=MixingConfig(mode="broyden", beta=0.7)
        ... )
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # 对角化
    # ═══════════════════════════════════════════════════════════════════════════

    davidson: Optional[DavidsonConfig] = Field(default=None, description="Davidson 对角化配置")

    # ═══════════════════════════════════════════════════════════════════════════
    # 密度混合
    # ═══════════════════════════════════════════════════════════════════════════

    mixing: Optional[MixingConfig] = Field(default=None, description="密度混合配置")

    # ═══════════════════════════════════════════════════════════════════════════
    # 性能
    # ═══════════════════════════════════════════════════════════════════════════

    use_gpu: bool = Field(default=True, description="是否使用 GPU 加速")

    gpu_id: int = Field(default=0, ge=0, description="GPU 设备 ID")

    # ═══════════════════════════════════════════════════════════════════════════
    # 诊断
    # ═══════════════════════════════════════════════════════════════════════════

    enable_diagnostics: bool = Field(default=False, description="是否启用诊断输出")

    diagnostics_dir: str = Field(default="./diagnostics", description="诊断输出目录")
