import xml.etree.ElementTree as ET

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_RY = 2.0


def parse_upf(filename):
    """
    Robust UPF v2 parser for ONCV and other XML-based pseudopotentials.
    Returns data in atomic units (Bohr for distance, Rydberg for potential).
    """
    tree = ET.parse(filename)
    root = tree.getroot()

    # 1. Mesh Information
    mesh_node = root.find(".//PP_MESH")
    if mesh_node is None:
        raise ValueError(f"PP_MESH not found in {filename}")

    r_node = mesh_node.find("PP_R")
    r_grid = np.fromstring(r_node.text, sep=" ")

    rab_node = mesh_node.find("PP_RAB")
    rab = np.fromstring(rab_node.text, sep=" ")

    # 2. Local Potential
    vloc_node = root.find(".//PP_LOCAL")
    if vloc_node is None:
        raise ValueError(f"PP_LOCAL not found in {filename}")
    vloc_r = np.fromstring(vloc_node.text, sep=" ")

    # 3. Non-local Projectors (Betas)
    betas = []
    l_list = []
    nl_node = root.find(".//PP_NONLOCAL")
    if nl_node is not None:
        # Sort by index if available to maintain consistency
        beta_nodes = sorted(
            [child for child in nl_node if child.tag.startswith("PP_BETA.")],
            key=lambda x: int(x.tag.split(".")[1]),
        )
        for child in beta_nodes:
            betas.append(np.fromstring(child.text, sep=" "))
            l_list.append(int(child.attrib.get("angular_momentum", 0)))

        # 4. Dij Matrix
        dij_node = nl_node.find("PP_DIJ")
        if dij_node is not None:
            dij_flat = np.fromstring(dij_node.text, sep=" ")
        else:
            dij_flat = np.array([])
    else:
        dij_flat = np.array([])

    # 5. Header information
    header = root.find(".//PP_HEADER")
    z_valence = float(header.attrib["z_valence"])

    # 6. Atomic Charge Density (for initial guess)
    rho_at_node = root.find(".//PP_RHOATOM")
    if rho_at_node is not None:
        rho_at_r = np.fromstring(rho_at_node.text, sep=" ")
    else:
        rho_at_r = np.array([])

    # 7. Core Charge Density (for NLCC)
    rho_core_node = root.find(".//PP_NLCC")
    if rho_core_node is not None:
        rho_core_r = np.fromstring(rho_core_node.text, sep=" ")
    else:
        rho_core_r = None

    # 8. Atomic Orbitals (CHI or fallback to BETA)
    chi_list = []
    chi_l_list = []

    # Try PP_CHI nodes first
    chi_nodes = root.findall(".//PP_CHI.*")
    if not chi_nodes:
        # Fallback to PP_PSWFC children if present
        pswfc_node = root.find(".//PP_PSWFC")
        if pswfc_node is not None:
            chi_nodes = [child for child in pswfc_node if child.tag.startswith("PP_CHI.")]

    if chi_nodes:
        # Sort and extract
        # Note: We need to handle index like PP_CHI.1
        try:
            sorted_nodes = sorted(chi_nodes, key=lambda x: int(x.tag.split(".")[-1]))
        except Exception:
            sorted_nodes = chi_nodes

        for child in sorted_nodes:
            chi_list.append(np.fromstring(child.text, sep=" "))
            chi_l_list.append(int(child.attrib.get("l", child.attrib.get("angular_momentum", 0))))

    return {
        "r": r_grid,
        "rab": rab,
        "vloc": vloc_r,  # Ry
        "betas": betas,
        "l_list": l_list,
        "dij": dij_flat,
        "zp": z_valence,
        "rho_at": rho_at_r,
        "rho_core": rho_core_r,
        "chi": chi_list,
        "chi_l": chi_l_list,
    }


def load_pseudo(filename, grid, atoms=None, type_idx=0):
    """
    Helper function to load a pseudopotential into dftcu objects.
    """
    import dftcu

    data = parse_upf(filename)

    vloc = dftcu.LocalPseudo(grid, atoms)
    # Note: init_tab_vloc expects vloc in Ry, r in Bohr, omega in Angstrom^3
    vloc.init_tab_vloc(type_idx, data["r"], data["vloc"], data["rab"], data["zp"], grid.volume())

    nl_pseudo = None
    if len(data["betas"]) > 0:
        nl_pseudo = dftcu.NonLocalPseudo(grid)
        nl_pseudo.init_tab_beta(
            type_idx, data["r"], data["betas"], data["rab"], data["l_list"], grid.volume()
        )
        nl_pseudo.init_dij(type_idx, data["dij"])

    return vloc, nl_pseudo, data
