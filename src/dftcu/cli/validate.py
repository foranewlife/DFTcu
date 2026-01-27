"""
DFTcu validate 命令

验证配置文件。
"""

import sys
from pathlib import Path

import click

from dftcu.config import DFTcuConfig


@click.command()
@click.argument(
    "config_file",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="显示详细信息",
)
def validate(config_file, verbose):
    """验证 YAML 配置文件

    \b
    示例:
        dftcu validate nscf_si.yaml
        dftcu validate scf_si.yaml --verbose
    """
    click.echo("=" * 70)
    click.echo("DFTcu 配置验证")
    click.echo("=" * 70)
    click.echo()

    click.echo(f"📋 验证配置文件: {config_file}")
    click.echo()

    try:
        # 加载配置
        config = DFTcuConfig.from_yaml(config_file)
        click.secho("✅ 配置文件格式正确", fg="green")
        click.echo()

        # 显示配置摘要
        click.echo("📊 配置摘要:")
        click.echo(f"  版本: {config.version}")
        if config.description:
            click.echo(f"  描述: {config.description}")
        click.echo()

        # 结构信息
        click.echo(f"结构文件: {config.structure_file}")
        click.echo()

        # Grid 配置 (数值离散)
        click.echo("Grid 配置 (Numerical Grid):")
        click.echo(f"  - FFT 网格: {config.grid.nr}")
        click.echo(
            f"  - ecutwfc: {config.grid.ecutwfc} eV = {config.grid.get_ecutwfc_hartree():.4f} Ha"
        )
        click.echo(
            f"  - ecutrho: {config.grid.ecutrho} eV = {config.grid.get_ecutrho_hartree():.4f} Ha"
        )
        click.echo(f"  - Gamma-only: {config.grid.is_gamma}")
        click.echo()

        # Physics 配置 (物理模型)
        click.echo("Physics 配置 (Physical Model):")
        atoms = config.physics.get_ase_atoms(config.structure_file)
        click.echo(f"  - 原子数: {len(atoms)}")
        click.echo(f"  - 电子数: {config.physics.nelec}")
        click.echo(f"  - 能带数: {config.physics.nbands}")
        click.echo(f"  - 占据方式: {config.physics.occupations}")
        if config.physics.smearing:
            click.echo(
                f"  - Smearing: {config.physics.smearing} (degauss={config.physics.degauss} eV)"
            )
        click.echo(f"  - XC 泛函: {config.physics.xc_functional}")
        click.echo()

        if verbose:
            click.echo("原子列表:")
            positions = atoms.get_positions()
            symbols = atoms.get_chemical_symbols()
            for i, (sym, pos) in enumerate(zip(symbols, positions)):
                click.echo(f"  {i+1}. {sym} @ {pos}")
            click.echo()

            click.echo("赝势列表:")
            for sym, filename in config.pseudopotentials.items():
                mass = config.get_mass(sym)
                click.echo(f"  - {sym}: {filename} (质量: {mass:.3f} amu)")
            click.echo()

        # Task 配置 (计算任务)
        click.echo("Task 配置 (Execution Task):")
        click.echo(f"  - 任务类型: {config.task.type}")
        click.echo(f"  - 输出目录: {config.task.outdir}")
        click.echo(f"  - 详细程度: {config.task.verbosity}")
        click.echo()

        if config.task.type == "scf" and config.task.scf:
            click.echo("SCF 细节:")
            click.echo(f"  - 最大迭代次数: {config.task.scf.max_iterations}")
            click.echo(f"  - 收敛阈值: {config.task.scf.conv_thr}")
            click.echo()

        if config.task.type == "nscf" and config.task.nscf:
            click.echo("NSCF 细节:")
            click.echo(f"  - 电荷密度文件: {config.task.nscf.charge_density_file or 'None'}")
            click.echo(f"  - 对角化方法: {config.task.nscf.diagonalization}")
            click.echo(f"  - 对角化阈值: {config.task.nscf.diago_thr}")
            click.echo()

        # Solver 配置 (数值求解器 - 可选)
        if config.solver:
            click.echo("Solver 配置 (Numerical Solver):")
            click.echo(f"  - 使用 GPU: {config.solver.use_gpu}")
            click.echo()

        # 验证晶格向量
        if verbose:
            click.echo("晶格向量 (Bohr):")
            lattice = config.grid.get_lattice_bohr(config.structure_file)
            for i, vec in enumerate(lattice):
                click.echo(f"  a{i+1} = [{vec[0]:8.4f}, {vec[1]:8.4f}, {vec[2]:8.4f}]")
            click.echo()

            click.echo("原子类型索引:")
            types = config.physics.get_atom_types(config.structure_file, config.pseudopotentials)
            click.echo(f"  {types}")
            click.echo()

            click.echo("赝势文件路径:")
            for sym in config.pseudopotentials.keys():
                pseudo_path = config.get_pseudo_path(sym)
                exists = pseudo_path.exists()
                status = "✅" if exists else "❌"
                click.echo(f"  {status} {sym}: {pseudo_path}")
            click.echo()

        click.secho("✅ 配置验证通过！", fg="green", bold=True)

    except Exception as e:
        click.secho(f"❌ 配置验证失败: {e}", fg="red", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)
