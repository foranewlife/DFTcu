#include <algorithm>
#include <fstream>
#include <sstream>

#include "upf_parser.cuh"
#include "utilities/error.cuh"

#include <pugixml.hpp>

namespace dftcu {

std::unique_ptr<PseudopotentialData> UPFParser::parse(const std::string& filename) {
    // 1. Load XML file
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(filename.c_str());

    if (!result) {
        throw std::runtime_error("Failed to parse UPF file: " + filename +
                                 " (Error: " + result.description() + ")");
    }

    auto data = std::make_unique<PseudopotentialData>();

    // 2. Get root node
    pugi::xml_node root = doc.child("UPF");
    if (!root) {
        throw std::runtime_error("Invalid UPF format: missing <UPF> root node in " + filename);
    }

    // 3. Parse sections
    parse_header(root, *data);
    parse_mesh(root, *data);
    parse_local(root, *data);
    parse_nonlocal(root, *data);

    // 4. Validate data
    if (!data->is_valid()) {
        throw std::runtime_error("Incomplete or invalid UPF data after parsing " + filename);
    }

    return data;
}

void UPFParser::parse_header(const pugi::xml_node& root, PseudopotentialData& data) {
    PseudopotentialHeader header;

    // UPF v2.0.0 format: <PP_HEADER> tag
    pugi::xml_node pp_header = root.child("PP_HEADER");
    if (!pp_header) {
        throw std::runtime_error("Missing PP_HEADER in UPF file");
    }

    // Read attributes
    header.element = trim(pp_header.attribute("element").as_string());
    header.pseudo_type = trim(pp_header.attribute("pseudo_type").as_string());
    header.functional = trim(pp_header.attribute("functional").as_string());
    header.z_valence = pp_header.attribute("z_valence").as_double();
    header.wfc_cutoff = pp_header.attribute("wfc_cutoff").as_double();
    header.rho_cutoff = pp_header.attribute("rho_cutoff").as_double();
    header.l_max = pp_header.attribute("l_max").as_int();
    header.l_local = pp_header.attribute("l_local").as_int();
    header.mesh_size = pp_header.attribute("mesh_size").as_int();
    header.number_of_proj = pp_header.attribute("number_of_proj").as_int();

    // Boolean attributes (UPF uses "T"/"F" strings)
    std::string is_us = trim(pp_header.attribute("is_ultrasoft").as_string());
    header.is_ultrasoft = (is_us == "T" || is_us == "true" || is_us == "TRUE");

    std::string is_paw_str = trim(pp_header.attribute("is_paw").as_string());
    header.is_paw = (is_paw_str == "T" || is_paw_str == "true" || is_paw_str == "TRUE");

    std::string core_corr = trim(pp_header.attribute("core_correction").as_string());
    header.core_correction = (core_corr == "T" || core_corr == "true" || core_corr == "TRUE");

    data.set_header(header);
}

void UPFParser::parse_mesh(const pugi::xml_node& root, PseudopotentialData& data) {
    RadialMesh mesh;

    pugi::xml_node pp_mesh = root.child("PP_MESH");
    if (!pp_mesh) {
        throw std::runtime_error("Missing PP_MESH in UPF file");
    }

    // Read mesh parameters
    mesh.dx = pp_mesh.attribute("dx").as_double();
    mesh.mesh = pp_mesh.attribute("mesh").as_int();
    mesh.xmin = pp_mesh.attribute("xmin").as_double();
    mesh.rmax = pp_mesh.attribute("rmax").as_double();
    mesh.zmesh = pp_mesh.attribute("zmesh").as_double(0.0);

    // Read PP_R (radial coordinates)
    pugi::xml_node pp_r = pp_mesh.child("PP_R");
    if (pp_r) {
        mesh.r = parse_numeric_array(pp_r);
    } else {
        throw std::runtime_error("Missing PP_R in PP_MESH");
    }

    // Read PP_RAB (integration weights)
    pugi::xml_node pp_rab = pp_mesh.child("PP_RAB");
    if (pp_rab) {
        mesh.rab = parse_numeric_array(pp_rab);
    } else {
        throw std::runtime_error("Missing PP_RAB in PP_MESH");
    }

    data.set_mesh(mesh);
}

void UPFParser::parse_local(const pugi::xml_node& root, PseudopotentialData& data) {
    LocalPotential local;

    pugi::xml_node pp_local = root.child("PP_LOCAL");
    if (!pp_local) {
        throw std::runtime_error("Missing PP_LOCAL in UPF file");
    }

    local.vloc_r = parse_numeric_array(pp_local);

    data.set_local(local);
}

void UPFParser::parse_nonlocal(const pugi::xml_node& root, PseudopotentialData& data) {
    NonlocalPotential nonlocal;

    pugi::xml_node pp_nonlocal = root.child("PP_NONLOCAL");
    if (!pp_nonlocal) {
        // Some local pseudopotentials may not have nonlocal part
        // This is valid for purely local pseudopotentials
        nonlocal.nbeta = 0;
        data.set_nonlocal(nonlocal);
        return;
    }

    // Parse all PP_BETA.* nodes
    for (pugi::xml_node pp_beta : pp_nonlocal.children()) {
        std::string node_name = pp_beta.name();

        // Match PP_BETA.1, PP_BETA.2, etc.
        if (node_name.find("PP_BETA") != std::string::npos) {
            BetaProjector beta;
            beta.index = pp_beta.attribute("index").as_int();
            beta.label = trim(pp_beta.attribute("label").as_string());
            beta.angular_momentum = pp_beta.attribute("angular_momentum").as_int();
            beta.cutoff_radius_index = pp_beta.attribute("cutoff_radius_index").as_int();
            beta.beta_r = parse_numeric_array(pp_beta);

            nonlocal.beta_functions.push_back(beta);
        }
    }

    nonlocal.nbeta = static_cast<int>(nonlocal.beta_functions.size());

    // Parse PP_DIJ matrix
    pugi::xml_node pp_dij = pp_nonlocal.child("PP_DIJ");
    if (pp_dij) {
        nonlocal.dij = parse_numeric_array(pp_dij);
    } else if (nonlocal.nbeta > 0) {
        throw std::runtime_error(
            "Missing PP_DIJ in PP_NONLOCAL (required when projectors present)");
    }

    data.set_nonlocal(nonlocal);
}

std::vector<double> UPFParser::parse_numeric_array(const pugi::xml_node& node) {
    std::vector<double> result;
    std::string text = node.text().get();

    std::istringstream iss(text);
    double value;
    while (iss >> value) {
        result.push_back(value);
    }

    return result;
}

std::string UPFParser::trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return "";
    }
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

std::string UPFParser::detect_version(const std::string& filename) {
    pugi::xml_document doc;
    if (!doc.load_file(filename.c_str())) {
        return "unknown";
    }

    pugi::xml_node root = doc.child("UPF");
    if (root) {
        return root.attribute("version").as_string("1.0.0");
    }

    return "unknown";
}

}  // namespace dftcu
