#pragma once

#include <string>
#include <vector>

#include "../model/grid.cuh"

namespace dftcu {

class Wavefunction;
class Hamiltonian;

struct VerificationResult {
    bool success;
    double h_sub_error;
    double s_sub_error;
};

class Phase0Verifier {
  public:
    Phase0Verifier(const Grid& grid);

    /**
     * Execute Phase 0 verification (S_sub only)
     * @param wfc_file Random wavefunction file path
     * @param s_ref_file QE S_sub reference data
     * @param nbands Number of bands
     * @param ecutwfc Wavefunction cutoff energy
     */
    VerificationResult verify(const std::string& wfc_file, const std::string& s_ref_file,
                              int nbands, double ecutwfc);

  private:
    const Grid& grid_;

    bool load_random_wavefunction(const std::string& filename, Wavefunction& psi);
    bool load_qe_s_reference(const std::string& s_file, std::vector<double>& S_ref, int& nbnd);

    double compute_matrix_error(const std::vector<double>& A, const std::vector<double>& B, int n);

    void print_matrix_comparison(const std::vector<double>& A, const std::vector<double>& B, int n,
                                 const std::string& name);
};

}  // namespace dftcu
