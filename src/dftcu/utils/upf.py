"""
UPF (Unified Pseudopotential Format) Parser for Python

负责解析 UPF v2 (XML) 文件并填充 C++ 的 PseudopotentialData 对象。
遵循设计原则：文件解析在 Python 层完成。
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import dftcu


class UPFParser:
    """Python 版 UPF 解析器"""

    def parse(self, filename: str | Path) -> dftcu.PseudopotentialData:
        """解析 UPF 文件并返回 C++ PseudopotentialData 对象"""
        tree = ET.parse(filename)
        root = tree.getroot()

        data = dftcu.PseudopotentialData()

        # 1. Header
        header = self._parse_header(root.find("PP_HEADER"))
        data.set_header(header)

        # 2. Mesh
        mesh = self._parse_mesh(root.find("PP_MESH"))
        data.set_mesh(mesh)

        # 3. Local Potential
        local = self._parse_local(root.find("PP_LOCAL"))
        data.set_local(local)

        # 4. Nonlocal Potential
        nonlocal_pot = self._parse_nonlocal(root.find("PP_NONLOCAL"))
        data.set_nonlocal(nonlocal_pot)

        # 5. Atomic Density
        rho_atom_node = root.find("PP_RHOATOM")
        if rho_atom_node is not None:
            atomic_density = self._parse_rho_atom(rho_atom_node)
            data.set_atomic_density(atomic_density)

        # 6. Atomic Orbitals (pswfc)
        pswfc_node = root.find("PP_PSWFC")
        if pswfc_node is not None:
            self._parse_pswfc(pswfc_node, data)

        return data

    def _parse_pswfc(self, node, data: dftcu.PseudopotentialData):
        aw = dftcu.AtomicWavefunctions()
        wavefunctions = []

        # Find all children that start with PP_CHI (some UPFs use PP_CHI.1, PP_CHI.2)
        for child in node:
            if child.tag.startswith("PP_CHI"):
                wfc = dftcu.PseudoAtomicWfc()
                attrs = child.attrib
                wfc.l = int(attrs.get("l", 0))
                wfc.label = attrs.get("label", "")
                wfc.occupation = float(attrs.get("occupation", 0.0))
                wfc.chi = [float(x) for x in child.text.split()]
                wavefunctions.append(wfc)

        aw.wavefunctions = wavefunctions
        data.set_atomic_wfc(aw)

    def _parse_header(self, node) -> dftcu.PseudopotentialHeader:
        h = dftcu.PseudopotentialHeader()
        attrs = node.attrib

        h.element = attrs.get("element", "")
        h.pseudo_type = attrs.get("pseudo_type", "")
        h.functional = attrs.get("functional", "")
        h.z_valence = float(attrs.get("z_valence", 0.0))
        h.wfc_cutoff = float(attrs.get("wfc_cutoff", 0.0))
        h.rho_cutoff = float(attrs.get("rho_cutoff", 0.0))
        h.l_max = int(attrs.get("l_max", 0))
        h.l_local = int(attrs.get("l_local", 0))
        h.mesh_size = int(attrs.get("mesh_size", 0))
        h.number_of_proj = int(attrs.get("number_of_proj", 0))
        h.is_ultrasoft = attrs.get("is_ultrasoft", "F").upper() == "T"
        h.is_paw = attrs.get("is_paw", "F").upper() == "T"
        h.core_correction = attrs.get("core_correction", "F").upper() == "T"

        return h

    def _parse_mesh(self, node) -> dftcu.RadialMesh:
        m = dftcu.RadialMesh()

        # r
        r_node = node.find("PP_R")
        if r_node is not None:
            m.r = [float(x) for x in r_node.text.split()]

        # rab
        rab_node = node.find("PP_RAB")
        if rab_node is not None:
            m.rab = [float(x) for x in rab_node.text.split()]

        attrs = node.attrib
        m.dx = float(attrs.get("dx", 0.0))
        m.xmin = float(attrs.get("xmin", 0.0))
        m.rmax = float(attrs.get("rmax", 0.0))
        m.mesh = int(attrs.get("mesh", 0))
        m.zmesh = float(attrs.get("zmesh", 0.0))

        return m

    def _parse_local(self, node) -> dftcu.LocalPotential:
        l = dftcu.LocalPotential()
        # V_LOC is usually the only child or has a specific tag
        # In UPF v2 it is <PP_LOCAL> text content or a child
        text = node.text.strip() if node.text else ""
        if not text:
            # Try children if any (sometimes it's wrapped)
            for child in node:
                text = child.text.strip()
                break

        l.vloc_r = [float(x) for x in text.split()]
        return l

    def _parse_nonlocal(self, node) -> dftcu.NonlocalPotential:
        nl = dftcu.NonlocalPotential()

        # Beta functions
        beta_functions = []
        for i, child in enumerate(node.findall("PP_BETA")):
            beta = dftcu.BetaProjector()
            attrs = child.attrib
            beta.index = int(attrs.get("index", i + 1))
            beta.angular_momentum = int(attrs.get("angular_momentum", 0))
            beta.cutoff_radius_index = int(attrs.get("cutoff_radius_index", 0))
            beta.label = attrs.get("label", "")
            beta.beta_r = [float(x) for x in child.text.split()]
            beta_functions.append(beta)

        nl.beta_functions = beta_functions
        nl.nbeta = len(beta_functions)

        # DIJ Matrix
        dij_node = node.find("PP_DIJ")
        if dij_node is not None:
            nl.dij = [float(x) for x in dij_node.text.split()]

        return nl

    def _parse_rho_atom(self, node) -> dftcu.AtomicDensity:
        ad = dftcu.AtomicDensity()
        ad.rho_at = [float(x) for x in node.text.split()]
        return ad
