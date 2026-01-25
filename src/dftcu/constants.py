"""
DFTcu 常量定义

提供单位转换常量。
"""

# ═══════════════════════════════════════════════════════════════════════════
# 单位转换常量
# ═══════════════════════════════════════════════════════════════════════════

# 长度单位
BOHR_TO_ANGSTROM = 0.529177210903  # 1 Bohr = 0.5292 Angstrom
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM  # 1 Angstrom = 1.8897 Bohr

# 能量单位
HA_TO_EV = 27.211386245988  # 1 Hartree = 27.2114 eV
EV_TO_HA = 1.0 / HA_TO_EV  # 1 eV = 0.03675 Hartree

HA_TO_RY = 2.0  # 1 Hartree = 2 Rydberg (exact)
RY_TO_HA = 0.5  # 1 Rydberg = 0.5 Hartree (exact)

# 质量单位
AMU_TO_AU = 1822.888486209  # 1 amu = 1822.888 a.u.
AU_TO_AMU = 1.0 / AMU_TO_AU

# 物理常数
PI = 3.141592653589793
TWO_PI = 2.0 * PI
