#include <cmath>

#include "pseudopotential_data.cuh"

namespace dftcu {

bool PseudopotentialData::is_valid() const {
    // Check Header
    if (header_.element.empty()) {
        return false;
    }

    if (header_.z_valence <= 0.0) {
        return false;
    }

    if (header_.mesh_size <= 0) {
        return false;
    }

    // Check Mesh
    if (mesh_.r.empty() || mesh_.rab.empty()) {
        return false;
    }

    if (mesh_.r.size() != static_cast<size_t>(header_.mesh_size)) {
        return false;
    }

    if (mesh_.rab.size() != static_cast<size_t>(header_.mesh_size)) {
        return false;
    }

    // Check Local Potential
    if (local_.vloc_r.empty()) {
        return false;
    }

    if (local_.vloc_r.size() != static_cast<size_t>(header_.mesh_size)) {
        return false;
    }

    // Check Nonlocal Potential (if present)
    if (header_.number_of_proj > 0) {
        if (nonlocal_.beta_functions.size() != static_cast<size_t>(header_.number_of_proj)) {
            return false;
        }

        int expected_dij_size = header_.number_of_proj * header_.number_of_proj;
        if (nonlocal_.dij.size() != static_cast<size_t>(expected_dij_size)) {
            return false;
        }

        // Verify each beta projector has correct mesh size
        for (const auto& beta : nonlocal_.beta_functions) {
            if (beta.beta_r.size() != static_cast<size_t>(header_.mesh_size)) {
                return false;
            }
        }
    }

    return true;
}

}  // namespace dftcu
