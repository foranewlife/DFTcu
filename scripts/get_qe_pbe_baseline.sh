#!/bin/bash
set -euo pipefail

# --- Find pw.x ---
QE_EXEC=""
QE_CANDIDATES=(
    "external/qe/build/bin/pw.x"
    "external/qe/bin/pw.x"
)
for candidate in "${QE_CANDIDATES[@]}"; do
    if [ -f "$candidate" ]; then
        QE_EXEC=$(realpath "$candidate")
        break
    fi
done

if [ -z "$QE_EXEC" ] && command -v pw.x &> /dev/null; then
    QE_EXEC="pw.x"
fi

LOCAL_QE_HINT="external/qe/build/bin/pw.x"
if [ -z "$QE_EXEC" ]; then
    echo "Error: pw.x not found locally ($LOCAL_QE_HINT) or in your PATH."
    echo "Attempting to compile QE..."
    (cd external/qe && ./coninfigure --without-libmbd && make pw -j
$(nproc))
    for candidate in "${QE_CANDIDATES[@]}"; do
        if [ -f "$candidate" ]; then
            QE_EXEC=$(realpath "$candidate")
            break
        fi
    done
    if [ -z "$QE_EXEC" ]; then
        echo "Compilation failed. Please install Quantum ESPRESSO manually."
        exit 1
    fi
fi
echo "Using pw.x executable: $QE_EXEC"


# --- Prepare run directory ---
WORKDIR="run_qe_pbe_final"
mkdir -p "$WORKDIR"

UPF_FILE="O_ONCV_PBE-1.2.upf"
UPF_SOURCE="external/QEpy/examples/jupyter/DATA/$UPF_FILE"
UPF_DEST="$WORKDIR/$UPF_FILE"

if [ ! -f "$UPF_DEST" ]; then
    cp -f "$UPF_SOURCE" "$UPF_DEST"
fi

ABS_WORKDIR=$(realpath "$WORKDIR")
mkdir -p "$ABS_WORKDIR/out"

# --- Create input file ---
cat > "$WORKDIR/qe.in" <<EOL
 &CONTROL
    calculation = 'scf',
    prefix = 'O',
    pseudo_dir = '$ABS_WORKDIR',
    outdir = '$ABS_WORKDIR/out'
 /
 &SYSTEM
    ibrav = 1, celldm(1) = 15.117, nat = 1, ntyp = 1,
    ecutwfc = 30.0, ecutrho = 120.0
 /
 &ELECTRONS
    conv_thr = 1.0d-8
 /
ATOMIC_SPECIES
 O  15.999  $UPF_FILE
ATOMIC_POSITIONS {angstrom}
 O  0.0  0.0  0.0
K_POINTS {gamma}
EOL

# --- Run Calculation ---
echo "Running QE SCF calculation..."
(cd "$WORKDIR" && "$QE_EXEC" -in qe.in > qe.out)

# --- Extract and Print Results ---
echo -e "\n--- QE Results ---"
grep "!    total energy" "$WORKDIR/qe.out" | tail -n 1
echo ""
echo "Kohn-Sham states (eV):"
grep -A 4 "Kohn-Sham states" "$WORKDIR/qe.out" | tail -n 4
