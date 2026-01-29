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
 * @brief chi(r) 径向轨道数据结构（用于 WavefunctionBuilder 测试）
 */
struct ChiRData {
    std::string element;  // 元素名（Si, C）
    std::string orbital;  // 轨道名（s, p, d, f）
    int l;                // 角动量量子数
    int mesh;             // 径向网格点数
    std::vector<double> r;    // 径向坐标 r (Bohr)
    std::vector<double> rab;  // 积分权重 dr (Bohr)
    std::vector<double> chi;  // chi(r) 值 (Bohr^(-3/2))
};

/**
 * @brief chi_q 表数据结构（用于 WavefunctionBuilder 测试）
 */
struct ChiQData {
    std::string element;  // 元素名（Si, C）
    std::string orbital;  // 轨道名（s, p, d, f）
    int l;                // 角动量量子数
    int nqx;              // q 点数量
    double dq;            // q 点间距 (Bohr^-1)
    std::vector<double> q;      // q 值 (Bohr^-1)
    std::vector<double> chi_q;  // chi_q 值 (Bohr^(3/2))
};

/**
 * @brief 原子波函数数据结构（用于 WavefunctionBuilder 测试）
 */
struct PsiAtomicData {
    std::string system;  // 体系名称（如 "SiC"）
    int ik;              // k 点索引
    int nbnd;            // band 数量
    int npw;             // G-vector 数量
    double omega;        // 晶胞体积 (Bohr^3)

    // 数据：[band][ig] -> (h, k, l, psi_re, psi_im, |psi|)
    struct DataPoint {
        int band;
        int ig;
        int h, k, l;  // Miller 指数
        double psi_re, psi_im, psi_abs;
    };
    std::vector<DataPoint> data;
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

    /**
     * @brief 加载 chi(r) 径向轨道数据
     * @param filename 文件路径（如 "chi_r_Si_s.dat"）
     * @return ChiRData 结构
     */
    static ChiRData load_chi_r(const std::string& filename) {
        ChiRData data;
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
                        while (!data.element.empty() && isspace(data.element.back()))
                            data.element.pop_back();
                    }
                } else if (line.find("orbital") != std::string::npos) {
                    size_t pos = line.find("=");
                    if (pos != std::string::npos) {
                        data.orbital = line.substr(pos + 2);
                        while (!data.orbital.empty() && isspace(data.orbital.back()))
                            data.orbital.pop_back();
                    }
                } else if (line.find("l =") != std::string::npos) {
                    sscanf(line.c_str(), "# l = %d", &data.l);
                } else if (line.find("mesh") != std::string::npos) {
                    sscanf(line.c_str(), "# mesh = %d", &data.mesh);
                }
                continue;
            }

            // 解析数据行: ir r rab chi
            int ir;
            double r_val, rab_val, chi_val;
            std::stringstream ss(line);
            if (ss >> ir >> r_val >> rab_val >> chi_val) {
                data.r.push_back(r_val);
                data.rab.push_back(rab_val);
                data.chi.push_back(chi_val);
            }
        }
        return data;
    }

    /**
     * @brief 加载 chi_q 表数据
     * @param filename 文件路径（如 "chi_q_Si_s.dat"）
     * @return ChiQData 结构
     */
    static ChiQData load_chi_q(const std::string& filename) {
        ChiQData data;
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
                } else if (line.find("orbital") != std::string::npos) {
                    size_t pos = line.find("=");
                    if (pos != std::string::npos) {
                        data.orbital = line.substr(pos + 2);
                        while (!data.orbital.empty() && isspace(data.orbital.back()))
                            data.orbital.pop_back();
                    }
                } else if (line.find("l =") != std::string::npos) {
                    sscanf(line.c_str(), "# l = %d", &data.l);
                } else if (line.find("nqx") != std::string::npos) {
                    sscanf(line.c_str(), "# nqx = %d", &data.nqx);
                } else if (line.find("dq") != std::string::npos) {
                    sscanf(line.c_str(), "# dq = %lf", &data.dq);
                }
                continue;
            }

            // 解析数据行: iq q chi_q
            int iq;
            double q_val, chi_q_val;
            std::stringstream ss(line);
            if (ss >> iq >> q_val >> chi_q_val) {
                data.q.push_back(q_val);
                data.chi_q.push_back(chi_q_val);
            }
        }
        return data;
    }

    /**
     * @brief 加载原子波函数数据
     * @param filename 文件路径（如 "psi_atomic_SiC.dat"）
     * @return PsiAtomicData 结构
     */
    static PsiAtomicData load_psi_atomic(const std::string& filename) {
        PsiAtomicData data;
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
                if (line.find("system") != std::string::npos) {
                    size_t pos = line.find("=");
                    if (pos != std::string::npos) {
                        data.system = line.substr(pos + 2);
                        while (!data.system.empty() && isspace(data.system.back()))
                            data.system.pop_back();
                    }
                } else if (line.find("ik =") != std::string::npos) {
                    sscanf(line.c_str(), "# ik = %d", &data.ik);
                } else if (line.find("nbnd") != std::string::npos) {
                    sscanf(line.c_str(), "# nbnd = %d", &data.nbnd);
                } else if (line.find("npw") != std::string::npos) {
                    sscanf(line.c_str(), "# npw = %d", &data.npw);
                } else if (line.find("omega") != std::string::npos) {
                    sscanf(line.c_str(), "# omega(Bohr^3) = %lf", &data.omega);
                }
                continue;
            }

            // 解析数据行: band ig h k l psi_re psi_im |psi|
            PsiAtomicData::DataPoint point;
            std::stringstream ss(line);
            if (ss >> point.band >> point.ig >> point.h >> point.k >> point.l >>
                     point.psi_re >> point.psi_im >> point.psi_abs) {
                data.data.push_back(point);
            }
        }
        return data;
    }
};

}  // namespace test
}  // namespace dftcu
