import dftcu
import numpy as np

# Create a grid
lattice = np.eye(3) * 10.0
nr = [32, 32, 32]
grid = dftcu.Grid(lattice.flatten().tolist(), nr)

print(f"Grid nnr: {grid.nnr()}")
print(f"Grid dv: {grid.dv()}")
print(f"Grid volume: {grid.volume()}")

# Create a field
rho = dftcu.RealField(grid, 1)
# Create a Gaussian density
rho_host = np.zeros(grid.nnr())
for i in range(nr[0]):
    for j in range(nr[1]):
        for k in range(nr[2]):
            idx = (i * nr[1] + j) * nr[2] + k
            dx = (i - nr[0] / 2) * lattice[0, 0] / nr[0]
            dy = (j - nr[1] / 2) * lattice[1, 1] / nr[1]
            dz = (k - nr[2] / 2) * lattice[2, 2] / nr[2]
            r2 = dx * dx + dy * dy + dz * dz
            rho_host[idx] = np.exp(-r2)
rho.copy_from_host(rho_host)

# Hartree potential
hartree = dftcu.Hartree(grid)
vh = dftcu.RealField(grid, 1)
energy = hartree.compute(rho, vh)

print(f"Hartree energy: {energy}")

# Copy back to host
vh_host = np.zeros(grid.nnr())
vh.copy_to_host(vh_host)
print(f"vh[0]: {vh_host[0]}")

# Atoms and Pseudo
atoms_list = [dftcu.Atom(0, 0, 0, 1.0, 0), dftcu.Atom(5, 5, 5, 1.0, 0)]
atoms = dftcu.Atoms(atoms_list)

pseudo = dftcu.LocalPseudo(grid, atoms)
vloc_g = np.ones(grid.nnr())  # Placeholder
pseudo.set_vloc(0, vloc_g.tolist())

vloc_r = dftcu.RealField(grid, 1)
pseudo.compute(vloc_r)

vloc_host = np.zeros(grid.nnr())
vloc_r.copy_to_host(vloc_host)
print(f"vloc[0]: {vloc_host[0]}")
