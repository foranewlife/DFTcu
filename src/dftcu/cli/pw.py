"""
DFTcu pw 命令

运行 DFT 计算（对标 QE pw.x）。
"""

import sys
from pathlib import Path

import click

from dftcu.config import DFTcuConfig


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="YAML 配置文件路径",
)
@click.option(
    "--outdir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="输出目录（覆盖配置文件中的设置）",
)
@click.option(
    "--validate-only",
    is_flag=True,
    help="仅验证配置，不运行计算",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="显示将要执行的操作，但不实际运行",
)
@click.pass_context
def pw(ctx, config, outdir, validate_only, dry_run):
    """运行 DFT 计算（SCF 或 NSCF）

    \b
    示例:
        dftcu pw --config nscf_si.yaml
        dftcu pw --config scf_si.yaml --outdir ./output
        dftcu pw --config nscf_si.yaml --validate-only
    """
    verbose = ctx.obj.get("verbose", 0)

    # 打印标题
    click.echo("=" * 70)
    click.echo("DFTcu - CUDA 加速的密度泛函理论计算")
    click.echo("=" * 70)
    click.echo()

    # 1. 加载配置
    click.echo(f"📋 加载配置文件: {config}")
    try:
        dftcu_config = DFTcuConfig.from_yaml(config)
        click.secho("✅ 配置加载成功", fg="green")
    except Exception as e:
        click.secho(f"❌ 配置加载失败: {e}", fg="red", err=True)
        sys.exit(1)

    # 覆盖输出目录
    if outdir:
        dftcu_config.task.outdir = str(outdir)
        click.echo(f"📁 输出目录: {outdir}")

    # 2. 显示配置摘要
    click.echo()
    click.echo("📊 配置摘要:")
    click.echo(f"  - 任务类型: {dftcu_config.task.type}")
    click.echo(f"  - 结构文件: {dftcu_config.structure_file}")
    click.echo(f"  - FFT 网格: {dftcu_config.grid.nr}")
    click.echo(f"  - 截断能: {dftcu_config.grid.ecutwfc} eV")
    click.echo(f"  - 电子数: {dftcu_config.physics.nelec}")
    click.echo(f"  - 能带数: {dftcu_config.physics.nbands}")
    click.echo(f"  - 输出目录: {dftcu_config.task.outdir}")

    # 3. 仅验证模式
    if validate_only:
        click.echo()
        click.secho("✅ 配置验证通过（--validate-only 模式）", fg="green")
        return

    # 4. Dry-run 模式
    if dry_run:
        click.echo()
        click.secho("🔍 Dry-run 模式：显示将要执行的操作", fg="yellow")
        click.echo()
        click.echo("将要执行的步骤:")
        click.echo("  1. 初始化 Grid (Bohr/Ha)")
        click.echo("  2. 初始化 Physics (Atoms/Potentials)")
        if dftcu_config.task.type == "scf":
            click.echo("  3. 运行 SCF 循环")
        else:
            click.echo("  3. 运行 NSCF 对角化")
        click.echo()
        click.secho("（实际计算未执行）", fg="yellow")
        return

    # 5. 运行计算
    click.echo()
    click.echo("=" * 70)
    click.echo("开始计算")
    click.echo("=" * 70)
    click.echo()

    try:
        if dftcu_config.task.type == "scf":
            _run_scf(dftcu_config, verbose)
        elif dftcu_config.task.type == "nscf":
            _run_nscf(dftcu_config, verbose)
        else:
            click.secho(f"❌ 不支持的任务类型: {dftcu_config.task.type}", fg="red", err=True)
            sys.exit(1)

        click.echo()
        click.secho("✅ 计算完成！", fg="green", bold=True)

    except Exception as e:
        click.echo()
        click.secho(f"❌ 计算失败: {e}", fg="red", err=True)
        if verbose > 0:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def _run_scf(config: DFTcuConfig, verbose: int):
    """运行 SCF 计算 (待实现)"""
    click.echo("🔄 SCF 计算流程")
    click.secho("⚠️  SCF 流程正在适配新架构，请改用 NSCF 进行测试。", fg="yellow")


def _run_nscf(config: DFTcuConfig, verbose: int):
    """运行 NSCF 计算 (按照 Brain-Heart 架构实现)"""
    click.echo("⚡ NSCF 计算流程 (Factory Mode)")

    try:
        import numpy as np

        import dftcu
        from dftcu.utils.upf import UPFParser as PythonUPFParser
    except ImportError as e:
        click.secho(f"❌ 无法导入模块: {e}", fg="red", err=True)
        sys.exit(1)

    # 1. 工厂生产基础模型 (Grid, Atoms)
    click.echo("  🏗️  正在初始化数值格点和原子结构...")
    lattice_bohr = config.grid.get_lattice_bohr(config.structure_file)
    grid = dftcu.create_grid_from_atomic_units(
        lattice_bohr,
        config.grid.nr,
        config.grid.get_ecutwfc_hartree(),
        config.grid.get_ecutrho_hartree(),
        config.grid.is_gamma,
    )

    ase_atoms = config.physics.get_ase_atoms(config.structure_file)
    symbols = ase_atoms.get_chemical_symbols()
    positions = ase_atoms.get_positions()
    unique_elements = list(config.pseudopotentials.keys())

    atoms_list = []
    for sym, pos in zip(symbols, positions):
        atom_type = unique_elements.index(sym)
        mass = config.get_mass(sym)
        atoms_list.append(dftcu.Atom(pos[0], pos[1], pos[2], mass, atom_type))

    atoms = dftcu.create_atoms_from_angstrom(atoms_list)
    click.echo(f"  ✅ Grid & Atoms 已就绪: {grid.nr()} 网格, {len(atoms_list)} 个原子")

    # 2. Python 解析并组装赝势数据
    click.echo("  📝 正在解析 UPF 赝势 (Python Parser)...")
    upf_parser = PythonUPFParser()
    pseudo_data_list = []
    for element in unique_elements:
        pseudo_path = config.get_pseudo_path(element)
        click.echo(f"    - 加载: {element} -> {pseudo_path.name}")
        data = upf_parser.parse(pseudo_path)
        pseudo_data_list.append(data)

    # 3. 组装 Hamiltonian (Python 侧)
    click.echo("  🛠️  正在组装哈密顿量对象...")
    dfp = dftcu.DensityFunctionalPotential(grid)
    dfp.add_functional(dftcu.Hartree())

    if config.physics.xc_functional.lower() == "lda":
        dfp.add_functional(dftcu.LDA_PZ())
    else:
        dfp.add_functional(dftcu.LDA_PZ())  # Fallback

    local_pseudos = []
    nonlocal_pseudos = []
    for i, data in enumerate(pseudo_data_list):
        lp = dftcu.create_local_pseudo(grid, atoms, data, i)
        dfp.add_functional(lp)
        local_pseudos.append(lp)

        nlp = dftcu.create_nonlocal_pseudo(grid, atoms, data, i)
        nonlocal_pseudos.append(nlp)

    ham = dftcu.Hamiltonian(grid)
    ham.set_density_functional_potential(dfp)
    if nonlocal_pseudos:
        ham.set_nonlocal(nonlocal_pseudos[0])

    # 4. 初始化密度 (原子电荷叠加)
    click.echo("  ⚛️  正在通过原子电荷叠加初始化电荷密度 (DensityFactory)...")
    density_factory = dftcu.DensityFactory(grid, atoms)
    for i, data in enumerate(pseudo_data_list):
        mesh = data.mesh()
        rho_at = data.atomic_density().rho_at
        if rho_at:
            density_factory.set_atomic_rho_r(i, mesh.r, rho_at, mesh.rab)

    rho = dftcu.RealField(grid)
    density_factory.build_density(rho)

    rho_data_host = np.zeros(grid.nnr())
    rho.copy_to_host(rho_data_host)
    click.echo(
        f"    - 初始密度积分: {np.sum(rho_data_host) * grid.dv():.4f} e (期望: {config.physics.nelec})"
    )

    # 5. 初始化波函数 (原子波函数叠加)
    click.echo("  🌊 正在通过原子波函数叠加初始化波函数 (WavefunctionFactory)...")
    wfc_factory = dftcu.WavefunctionFactory(grid, atoms)
    for i, data in enumerate(pseudo_data_list):
        mesh = data.mesh()
        for wfc in data.atomic_wfc().wavefunctions:
            wfc_factory.add_atomic_orbital(i, wfc.l, mesh.r, wfc.chi, mesh.rab)

    psi = dftcu.Wavefunction(grid, config.physics.nbands, config.grid.get_ecutwfc_hartree())
    wfc_factory.build_atomic_wavefunctions(psi, randomize_phase=False)
    psi.orthonormalize()
    click.echo(f"    - 波函数已初始化并归一化 ({config.physics.nbands} bands)")

    # 6. 调用 Workflow 运行流程
    click.echo("  🚀 正在启动 NSCF 工作流...")

    # 确保输出目录存在 (支持 C++ 端的 debug 导出)
    output_path = Path(config.task.outdir)
    output_path.mkdir(parents=True, exist_ok=True)
    # 特别是 nscf_output (一些 C++ debug 硬编码了此路径)
    Path("nscf_output").mkdir(parents=True, exist_ok=True)

    wf_config = dftcu.NSCFWorkflowConfig()
    wf_config.nbands = config.physics.nbands
    wf_config.nelec = config.physics.nelec
    wf_config.enable_diagnostics = config.task.verbosity == "high"
    wf_config.output_dir = config.task.outdir

    workflow = dftcu.NSCFWorkflow(grid, atoms, ham, psi, rho_data_host.tolist(), wf_config)
    result = workflow.execute()

    # 7. 汇报结果
    click.echo()
    click.secho("🏁 NSCF 计算完成!", fg="green", bold=True)
    click.echo(f"  总能量: {result.etot:16.10f} Ha")
    click.echo(f"  总能量: {result.etot * 27.211386245988:16.10f} eV")
    click.echo()
    click.echo("  本征值 (Ha):")
    for i, e in enumerate(result.eigenvalues):
        click.echo(f"    Band {i+1:2d}: {e:16.10f} Ha ({e * 27.211386245988:12.6f} eV)")
