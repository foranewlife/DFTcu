import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import dftcu


def _require_qepy():
    try:
        from qepy.driver import Driver  # noqa: F401
    except ImportError as exc:  # pragma: no cover - import guard
        pytest.skip(f"QEpy not available: {exc}")
    return Driver


def _prepare_qe_inputs(work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    upf_name = "O_ONCV_PBE-1.2.upf"
    pseudo_src = Path("external") / "QEpy" / "examples" / "jupyter" / "DATA" / upf_name
    if not pseudo_src.exists():
        pytest.skip(f"Pseudopotential not found at {pseudo_src}")
    pseudo_dst = work_dir / upf_name
    if not pseudo_dst.exists():
        shutil.copy(pseudo_src, pseudo_dst)

    qe_input = f"""
 &CONTROL
    calculation = 'scf', pseudo_dir = '{work_dir.resolve()}',
    outdir = '{(work_dir / 'out').resolve()}', prefix = 'pwscf'
 /
 &SYSTEM
    ibrav = 1, celldm(1) = 10.0, nat = 1, ntyp = 1,
    ecutwfc = 30.0
 /
 &ELECTRONS
    conv_thr = 1.0d-10
 /
ATOMIC_SPECIES
 O  15.999  {upf_name}
ATOMIC_POSITIONS {{bohr}}
 O  0.0  0.0  0.0
K_POINTS {{gamma}}
"""
    input_path = work_dir / "qe_live.in"
    with open(input_path, "w", encoding="utf-8") as fh:
        fh.write(qe_input)
    (work_dir / "out").mkdir(exist_ok=True)
    return input_path


def _run_qe_single_point(work_dir: Path):
    driver_class = _require_qepy()
    input_path = _prepare_qe_inputs(work_dir)

    old_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        driver = driver_class(str(input_path.name), iterative=True)
        driver.diagonalize()

        rho = driver.get_density(gather=True)
        # Collapse spin dimension if present
        if rho.ndim == 2:
            rho_tot = rho.sum(axis=1)
        else:
            rho_tot = rho

        evc = np.array(driver.get_wave_function(), dtype=np.complex128)
        occ = np.array(driver.get_occupation_numbers(), dtype=np.float64)
        driver.stop()
    finally:
        os.chdir(old_cwd)

    return rho_tot, evc, occ


@pytest.mark.qe_align
def test_qe_generated_data_matches_dftcu_density():
    """Run QE via QEpy, then verify DFTcu reproduces the same density."""
    rho_qe, evc_qe, occupations = _run_qe_single_point(Path("run_qe_align_live"))

    if evc_qe.ndim != 2:
        pytest.skip("Unexpected wavefunction array shape from QEpy.")

    nr = [36, 36, 36]
    grid = dftcu.Grid((np.eye(3) * 10.0).flatten().tolist(), nr)
    num_bands = evc_qe.shape[0]
    wf = dftcu.Wavefunction(grid, num_bands, encut=15.0)
    wf.copy_from_host(evc_qe.flatten())

    rho_field = dftcu.RealField(grid)
    occ_list = occupations[:num_bands].tolist()
    wf.compute_density(occ_list, rho_field)

    rho_dftcu = np.zeros(grid.nnr())
    rho_field.copy_to_host(rho_dftcu)

    if rho_qe.size != rho_dftcu.size:
        pytest.skip("QE grid points mismatch this DFTcu configuration.")

    diff = np.max(np.abs(rho_dftcu - rho_qe))
    print(f"[QE Live Alignment] max|rho_dftcu - rho_qe| = {diff:.3e}")
    assert diff < 1e-12

    total_charge_qe = rho_qe.sum() * grid.dv()
    assert abs(rho_field.integral() - total_charge_qe) < 1e-10
