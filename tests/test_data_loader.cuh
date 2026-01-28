/**
 * @file test_data_loader.cuh
 * @brief 测试数据加载工具：加载带物理索引的 QE 参考数据
 */

#pragma once

#include <complex>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// 前向声明
namespace dftcu {
class Grid;
}

namespace dftcu {
namespace test {

/**
 * @brief 加载带物理索引的 G 空间数据
 *
 * 格式：band h k l Re Im
 * 返回：map[(band, h, k, l)] = complex
 */
inline std::map<std::tuple<int, int, int, int>, std::complex<double>> load_g_space_data(
    const std::string& filename) {
    std::map<std::tuple<int, int, int, int>, std::complex<double>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        if (line.empty() || line[0] == '#')
            continue;

        int band, h, k, l;
        double re, im;
        std::istringstream iss(line);
        if (iss >> band >> h >> k >> l >> re >> im) {
            auto key = std::make_tuple(band, h, k, l);
            data[key] = std::complex<double>(re, im);
        } else {
            throw std::runtime_error("Parse error at line " + std::to_string(line_num) +
                                     " in file: " + filename);
        }
    }

    if (data.empty()) {
        throw std::runtime_error("No data loaded from file: " + filename);
    }

    return data;
}

/**
 * @brief 加载带物理索引的实空间数据
 *
 * 格式：ir ix iy iz Re Im
 * 返回：map[(ix, iy, iz)] = complex
 */
inline std::map<std::tuple<int, int, int>, std::complex<double>> load_r_space_data(
    const std::string& filename) {
    std::map<std::tuple<int, int, int>, std::complex<double>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        if (line.empty() || line[0] == '#')
            continue;

        int ir, ix, iy, iz;
        double re, im;
        std::istringstream iss(line);
        if (iss >> ir >> ix >> iy >> iz >> re >> im) {
            auto key = std::make_tuple(ix, iy, iz);
            data[key] = std::complex<double>(re, im);
        } else {
            throw std::runtime_error("Parse error at line " + std::to_string(line_num) +
                                     " in file: " + filename);
        }
    }

    if (data.empty()) {
        throw std::runtime_error("No data loaded from file: " + filename);
    }

    return data;
}

/**
 * @brief 加载能量标量
 */
inline double load_energy(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        try {
            return std::stod(line);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to parse energy from: " + filename);
        }
    }

    throw std::runtime_error("No energy value found in file: " + filename);
}

/**
 * @brief 加载实空间标量场（带物理索引）
 *
 * 格式：ix iy iz value
 * 返回：map[(ix, iy, iz)] = value
 */
inline std::map<std::tuple<int, int, int>, double> load_real_field_data(
    const std::string& filename) {
    std::map<std::tuple<int, int, int>, double> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        if (line.empty() || line[0] == '#')
            continue;

        int ix, iy, iz;
        double value;
        std::istringstream iss(line);
        if (iss >> ix >> iy >> iz >> value) {
            auto key = std::make_tuple(ix, iy, iz);
            data[key] = value;
        } else {
            throw std::runtime_error("Parse error at line " + std::to_string(line_num) +
                                     " in file: " + filename);
        }
    }

    if (data.empty()) {
        throw std::runtime_error("No data loaded from file: " + filename);
    }

    return data;
}

/**
 * @brief 加载实空间标量场（仅线性索引）
 *
 * 格式：ir value
 * 返回：vector[ir] = value
 */
inline std::vector<double> load_real_field_linear(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        if (line.empty() || line[0] == '#')
            continue;

        int ir;
        double value;
        std::istringstream iss(line);
        if (iss >> ir >> value) {
            // 确保 vector 足够大（ir 是 1-based）
            if (ir > static_cast<int>(data.size())) {
                data.resize(ir);
            }
            data[ir - 1] = value;  // 转换为 0-based
        } else {
            throw std::runtime_error("Parse error at line " + std::to_string(line_num) +
                                     " in file: " + filename);
        }
    }

    if (data.empty()) {
        throw std::runtime_error("No data loaded from file: " + filename);
    }

    return data;
}

}  // namespace test
}  // namespace dftcu
