#pragma once
#include <string>
#include <vector>

namespace dftcu {

/**
 * @brief UPF header information
 */
struct PseudopotentialHeader {
    std::string element;      ///< Element symbol (e.g., "Si")
    std::string pseudo_type;  ///< "NC" / "USPP" / "PAW"
    std::string functional;   ///< "LDA" / "PBE" / "PW91"
    double z_valence;         ///< Number of valence electrons
    double wfc_cutoff;        ///< Wavefunction cutoff energy (Ry)
    double rho_cutoff;        ///< Charge density cutoff energy (Ry)
    int l_max;                ///< Maximum angular momentum
    int l_local;              ///< Angular momentum for local potential
    int mesh_size;            ///< Number of radial mesh points
    int number_of_proj;       ///< Number of projectors
    bool is_ultrasoft;        ///< True if ultrasoft pseudopotential
    bool is_paw;              ///< True if PAW pseudopotential
    bool core_correction;     ///< True if nonlinear core correction

    PseudopotentialHeader()
        : z_valence(0.0),
          wfc_cutoff(0.0),
          rho_cutoff(0.0),
          l_max(0),
          l_local(0),
          mesh_size(0),
          number_of_proj(0),
          is_ultrasoft(false),
          is_paw(false),
          core_correction(false) {}
};

/**
 * @brief Radial mesh information
 */
struct RadialMesh {
    std::vector<double> r;    ///< Radial coordinates r (Bohr)
    std::vector<double> rab;  ///< Integration weights dr/dx
    double dx;                ///< Logarithmic mesh spacing
    double xmin;              ///< Minimum x = log(r)
    double rmax;              ///< Maximum radius (Bohr)
    int mesh;                 ///< Number of mesh points
    double zmesh;             ///< Nuclear charge for mesh generation

    RadialMesh() : dx(0.0), xmin(0.0), rmax(0.0), mesh(0), zmesh(0.0) {}
};

/**
 * @brief Local pseudopotential data
 */
struct LocalPotential {
    std::vector<double> vloc_r;  ///< V_loc(r) on radial mesh (Ry)

    LocalPotential() = default;
};

/**
 * @brief Beta projector function
 */
struct BetaProjector {
    int index;                   ///< Projector index (1-based)
    std::string label;           ///< Label (e.g., "3S", "3P")
    int angular_momentum;        ///< Angular momentum l
    int cutoff_radius_index;     ///< Index of cutoff radius in mesh
    std::vector<double> beta_r;  ///< β(r) on radial mesh

    BetaProjector() : index(0), angular_momentum(0), cutoff_radius_index(0) {}
};

/**
 * @brief Nonlocal pseudopotential data
 */
struct NonlocalPotential {
    std::vector<BetaProjector> beta_functions;  ///< All projectors
    std::vector<double> dij;                    ///< D_ij matrix (flattened, row-major)
    int nbeta;                                  ///< Number of projectors

    NonlocalPotential() : nbeta(0) {}
};

/**
 * @brief Complete pseudopotential data parsed from UPF file
 *
 * This class stores all data extracted from a UPF (Unified Pseudopotential Format) file.
 * It provides a structured interface for accessing pseudopotential information needed
 * by LocalPseudo and NonlocalPseudo classes.
 */
class PseudopotentialData {
  public:
    PseudopotentialData() = default;

    // Getter methods
    const PseudopotentialHeader& header() const { return header_; }
    const RadialMesh& mesh() const { return mesh_; }
    const LocalPotential& local() const { return local_; }
    const NonlocalPotential& nonlocal() const { return nonlocal_; }

    // Setter methods (for parser use)
    void set_header(const PseudopotentialHeader& h) { header_ = h; }
    void set_mesh(const RadialMesh& m) { mesh_ = m; }
    void set_local(const LocalPotential& l) { local_ = l; }
    void set_nonlocal(const NonlocalPotential& nl) { nonlocal_ = nl; }

    // Convenience accessors
    std::string element() const { return header_.element; }
    double z_valence() const { return header_.z_valence; }
    int mesh_size() const { return header_.mesh_size; }
    int number_of_proj() const { return header_.number_of_proj; }
    std::string pseudo_type() const { return header_.pseudo_type; }
    std::string functional() const { return header_.functional; }

    /**
     * @brief Validate data completeness and consistency
     * @return true if all required data is present and consistent
     */
    bool is_valid() const;

  private:
    PseudopotentialHeader header_;
    RadialMesh mesh_;
    LocalPotential local_;
    NonlocalPotential nonlocal_;
};

}  // namespace dftcu
