#pragma once
#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dftcu {
namespace test {

struct IndexedRealValue {
    int ix, iy, iz;
    double value;
};

struct IndexedComplexValue {
    int h, k, l;
    std::complex<double> value;
};

/**
 * @brief QE vloc_tab 数据结构
 */
struct VlocTabData {
    int nqx;
    double dq;
    int atom_type;
    std::string element;
    double zp;
    double omega;
    std::vector<double> tab;  // tab[iq], iq = 0..nqx
};

/**
 * @brief UPF 径向网格数据结构（用于 mock UPF）
 */
struct UPFRadialData {
    std::string element;
    int msh;                     // 有效网格点数
    double zp;                   // 价电子数
    double omega;                // 晶胞体积 (Bohr^3)
    std::vector<double> r;       // 径向坐标 r (Bohr)
    std::vector<double> rab;     // 积分权重 dr
    std::vector<double> vloc_r;  // V_loc(r) (Ry)
};

/**
 * @brief Utility to load indexed reference data.
 */
class StandardDataLoader {
  public:
    /**
     * @brief 加载 QE vloc_tab 数据文件
     * @param filename 文件路径
     * @return VlocTabData 结构
     */
    static VlocTabData load_vloc_tab(const std::string& filename) {
        VlocTabData data;
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty())
                continue;

            if (line[0] == '#') {
                // 解析元数据
                if (line.find("nqx") != std::string::npos) {
                    sscanf(line.c_str(), "# nqx = %d", &data.nqx);
                } else if (line.find("dq") != std::string::npos) {
                    sscanf(line.c_str(), "# dq = %lf", &data.dq);
                } else if (line.find("atom_type") != std::string::npos) {
                    sscanf(line.c_str(), "# atom_type = %d", &data.atom_type);
                } else if (line.find("element") != std::string::npos) {
                    size_t pos = line.find("=");
                    if (pos != std::string::npos) {
                        data.element = line.substr(pos + 2);
                        // trim whitespace
                        while (!data.element.empty() && isspace(data.element.back()))
                            data.element.pop_back();
                    }
                } else if (line.find("zp") != std::string::npos) {
                    sscanf(line.c_str(), "# zp = %lf", &data.zp);
                } else if (line.find("omega") != std::string::npos) {
                    sscanf(line.c_str(), "# omega = %lf", &data.omega);
                }
                continue;
            }

            // 解析数据行: iq value
            int iq;
            double value;
            std::stringstream ss(line);
            if (ss >> iq >> value) {
                if (iq >= static_cast<int>(data.tab.size())) {
                    data.tab.resize(iq + 1);
                }
                data.tab[iq] = value;
            }
        }
        return data;
    }

    /**
     * @brief 加载 V_ps(r) 实空间势数据
     * @param filename 文件路径
     * @return vector of values indexed by ir
     */
    static std::vector<double> load_v_ps_r(const std::string& filename) {
        std::vector<double> data;
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::string line;
        int nnr = 0;
        while (std::getline(file, line)) {
            if (line.empty())
                continue;

            if (line[0] == '#') {
                if (line.find("nnr") != std::string::npos) {
                    sscanf(line.c_str(), "# nnr = %d", &nnr);
                    data.reserve(nnr);
                }
                continue;
            }

            int ir;
            double value;
            std::stringstream ss(line);
            if (ss >> ir >> value) {
                // ir is 1-based in QE output
                if (ir > static_cast<int>(data.size())) {
                    data.resize(ir);
                }
                data[ir - 1] = value;  // Convert to 0-based
            }
        }
        return data;
    }

    /**
     * @brief 加载 V_loc(G) 复数数据
     * @param filename 文件路径
     * @return vector of complex values indexed by ig
     */
    static std::vector<std::complex<double>> load_vloc_g(const std::string& filename) {
        std::vector<std::complex<double>> data;
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#')
                continue;

            int ig;
            double re, im, abs_val;
            std::stringstream ss(line);
            if (ss >> ig >> re >> im >> abs_val) {
                if (ig >= static_cast<int>(data.size())) {
                    data.resize(ig + 1);
                }
                data[ig] = std::complex<double>(re, im);
            }
        }
        return data;
    }

    static std::vector<IndexedRealValue> load_real(const std::string& filename) {
        std::vector<IndexedRealValue> data;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open reference file: " << filename << std::endl;
            return data;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            std::stringstream ss(line);
            IndexedRealValue val;
            if (ss >> val.ix >> val.iy >> val.iz >> val.value) {
                data.push_back(val);
            }
        }
        return data;
    }

    static std::vector<IndexedComplexValue> load_complex(const std::string& filename) {
        std::vector<IndexedComplexValue> data;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open reference file: " << filename << std::endl;
            return data;
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#')
                continue;
            std::stringstream ss(line);
            IndexedComplexValue val;
            double re, im;
            if (ss >> val.h >> val.k >> val.l >> re >> im) {
                val.value = std::complex<double>(re, im);
                data.push_back(val);
            }
        }
        return data;
    }

    /**
     * @brief 加载 UPF 径向网格数据（用于 mock UPF）
     * @param filename 文件路径
     * @return UPFRadialData 结构
     */
    static UPFRadialData load_upf_radial(const std::string& filename) {
        UPFRadialData data;
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty())
                continue;

            if (line[0] == '#') {
                // 解析元数据
                if (line.find("element") != std::string::npos) {
                    size_t pos = line.find("=");
                    if (pos != std::string::npos) {
                        data.element = line.substr(pos + 2);
                        // trim whitespace
                        while (!data.element.empty() && isspace(data.element.back()))
                            data.element.pop_back();
                    }
                } else if (line.find("msh") != std::string::npos) {
                    sscanf(line.c_str(), "# msh = %d", &data.msh);
                } else if (line.find("zp") != std::string::npos) {
                    sscanf(line.c_str(), "# zp = %lf", &data.zp);
                } else if (line.find("omega") != std::string::npos) {
                    sscanf(line.c_str(), "# omega = %lf", &data.omega);
                }
                continue;
            }

            // 解析数据行: ir r(ir) rab(ir) vloc(ir)
            int ir;
            double r_val, rab_val, vloc_val;
            std::stringstream ss(line);
            if (ss >> ir >> r_val >> rab_val >> vloc_val) {
                // ir is 1-based, convert to 0-based
                if (ir > static_cast<int>(data.r.size())) {
                    data.r.resize(ir);
                    data.rab.resize(ir);
                    data.vloc_r.resize(ir);
                }
                data.r[ir - 1] = r_val;
                data.rab[ir - 1] = rab_val;
                data.vloc_r[ir - 1] = vloc_val;
            }
        }
        return data;
    }
};

}  // namespace test
}  // namespace dftcu
