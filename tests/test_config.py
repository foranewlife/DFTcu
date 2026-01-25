"""
配置验证测试

测试 Pydantic 配置模型的验证功能。
"""

import pytest

from dftcu.config import DFTcuConfig, GridConfig, PhysicsConfig, TaskConfig, TaskType


@pytest.fixture
def dummy_poscar(tmp_path):
    """创建临时 POSCAR 文件"""
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        """Si FCC structure
1.0
  -5.1  0.0  5.1
   0.0  5.1  5.1
  -5.1  5.1  0.0
Si
2
Direct
  0.00  0.00  0.00
  0.25  0.25  0.25
"""
    )
    return str(poscar)


class TestGridConfig:
    """Grid 配置测试"""

    def test_basic_grid(self):
        """测试基本 Grid 配置"""
        config = GridConfig(nr=[15, 15, 15], ecutwfc=163.268, ecutrho=653.072, is_gamma=True)
        assert config.nr == [15, 15, 15]


class TestPhysicsConfig:
    """Physics 配置测试"""

    def test_basic_physics(self, dummy_poscar):
        """测试基本 Physics 配置"""
        config = PhysicsConfig(nelec=8.0, nbands=4, occupations="fixed")
        assert config.nelec == 8.0
        # 验证原子读取
        atoms = config.get_ase_atoms(dummy_poscar)
        assert len(atoms) == 2

    def test_atom_types(self, dummy_poscar):
        """测试原子类型索引"""
        config = PhysicsConfig(nelec=8.0, nbands=4)
        pseudos = {"Si": "Si.UPF"}
        types = config.get_atom_types(dummy_poscar, pseudos)
        assert types == [0, 0]


class TestTaskConfig:
    """Task 配置测试"""

    def test_scf_task(self):
        """测试 SCF 任务配置"""
        config = TaskConfig(type="scf")
        assert config.type == TaskType.SCF
        assert config.scf is not None

    def test_nscf_task(self):
        """测试 NSCF 任务配置"""
        config = TaskConfig(type="nscf")
        assert config.type == TaskType.NSCF
        assert config.nscf is not None


class TestDFTcuConfig:
    """顶层配置测试"""

    def test_full_config(self, dummy_poscar):
        """测试完整配置"""
        config = DFTcuConfig(
            version="1.0.0",
            structure_file=dummy_poscar,
            pseudopotentials={"Si": "Si.UPF"},
            grid=GridConfig(
                nr=[15, 15, 15],
                ecutwfc=163.0,
            ),
            physics=PhysicsConfig(
                nelec=8.0,
                nbands=4,
            ),
            task=TaskConfig(
                type="scf",
            ),
        )
        assert config.structure_file == dummy_poscar
        assert config.pseudopotentials["Si"] == "Si.UPF"
        assert config.physics.nelec == 8.0
        assert config.task.type == TaskType.SCF

    def test_yaml_roundtrip(self, dummy_poscar, tmp_path):
        """测试 YAML 读写"""
        config = DFTcuConfig(
            version="1.0.0",
            structure_file=dummy_poscar,
            pseudopotentials={"Si": "Si.UPF"},
            grid=GridConfig(
                nr=[15, 15, 15],
                ecutwfc=163.0,
            ),
            physics=PhysicsConfig(
                nelec=8.0,
                nbands=4,
            ),
            task=TaskConfig(type="scf"),
        )

        yaml_file = tmp_path / "test.yaml"
        config.to_yaml(yaml_file)

        config2 = DFTcuConfig.from_yaml(yaml_file)
        assert config2.physics.nelec == config.physics.nelec
        assert config2.task.type == config.task.type
