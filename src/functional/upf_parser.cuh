#pragma once
#include <memory>
#include <string>

#include "pseudopotential_data.cuh"

// Forward declaration for pugixml
namespace pugi {
class xml_node;
}

namespace dftcu {

/**
 * @brief Parser for Unified Pseudopotential Format (UPF) files
 *
 * Supports UPF v2.0.0 format used by Quantum ESPRESSO.
 * Parses XML structure and populates PseudopotentialData.
 *
 * Example usage:
 * @code
 *   UPFParser parser;
 *   auto upf_data = parser.parse("Si.pz-rrkj.UPF");
 *   std::cout << "Element: " << upf_data->element() << std::endl;
 *   std::cout << "Z_valence: " << upf_data->z_valence() << std::endl;
 * @endcode
 */
class UPFParser {
  public:
    UPFParser() = default;

    /**
     * @brief Parse UPF file and extract pseudopotential data
     * @param filename Path to UPF file
     * @return Unique pointer to parsed pseudopotential data
     * @throws std::runtime_error if file not found or invalid format
     */
    std::unique_ptr<PseudopotentialData> parse(const std::string& filename);

    /**
     * @brief Detect UPF version from file
     * @param filename Path to UPF file
     * @return Version string (e.g., "2.0.0", "1.0.0")
     */
    static std::string detect_version(const std::string& filename);

  private:
    // Section parsers
    void parse_header(const pugi::xml_node& root, PseudopotentialData& data);
    void parse_mesh(const pugi::xml_node& root, PseudopotentialData& data);
    void parse_local(const pugi::xml_node& root, PseudopotentialData& data);
    void parse_nonlocal(const pugi::xml_node& root, PseudopotentialData& data);

    // Utility methods
    std::vector<double> parse_numeric_array(const pugi::xml_node& node);
    std::string trim(const std::string& str);
};

}  // namespace dftcu
