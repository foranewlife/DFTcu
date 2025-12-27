#pragma once

#include "error.cuh"
#include "gpu_macro.cuh"

/**
 * @brief CUDA kernel to fill a device array with a constant value.
 * @tparam T Data type of the array elements.
 * @param size Number of elements in the array.
 * @param value The value to fill the array with.
 * @param data Pointer to the device memory.
 */
template <typename T>
static void __global__ gpu_fill(const size_t size, const T value, T* data) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        data[i] = value;
}

/**
 * @brief Types of GPU memory allocations.
 */
enum class Memory_Type {
    global = 0, /**< Standard device memory (gpuMalloc) */
    managed     /**< Unified Memory (gpuMallocManaged), accessible from both CPU and GPU */
};

/**
 * @brief A lightweight container for GPU-resident vectors.
 *
 * Provides basic RAII management for CUDA memory, move semantics,
 * and convenient data transfer between host and device.
 *
 * @tparam T Element type (e.g., double, cuDoubleComplex).
 */
template <typename T>
class GPU_Vector {
  public:
    /**
     * @brief Constructs an empty GPU_Vector.
     */
    GPU_Vector() {
        size_ = 0;
        memory_ = 0;
        memory_type_ = Memory_Type::global;
        allocated_ = false;
    }

    /**
     * @brief Constructs a GPU_Vector and allocates memory.
     * @param size Number of elements to allocate.
     * @param memory_type Type of memory (global or managed).
     */
    GPU_Vector(const size_t size, const Memory_Type memory_type = Memory_Type::global) {
        allocated_ = false;
        resize(size, memory_type);
    }

    /**
     * @brief Destructor. Automatically frees GPU memory.
     */
    ~GPU_Vector() {
        if (allocated_) {
            CHECK(gpuFree(data_));
            allocated_ = false;
        }
    }

    // Disable copy to prevent accidental expensive transfers or double-frees
    GPU_Vector(const GPU_Vector&) = delete;
    GPU_Vector& operator=(const GPU_Vector&) = delete;

    /**
     * @brief Move constructor. Transfers ownership of the GPU memory.
     */
    GPU_Vector(GPU_Vector&& other) noexcept
        : allocated_(other.allocated_),
          size_(other.size_),
          memory_(other.memory_),
          memory_type_(other.memory_type_),
          data_(other.data_) {
        other.allocated_ = false;
        other.data_ = nullptr;
        other.size_ = 0;
    }

    /**
     * @brief Move assignment operator. Transfers ownership of the GPU memory.
     */
    GPU_Vector& operator=(GPU_Vector&& other) noexcept {
        if (this != &other) {
            if (allocated_)
                CHECK(gpuFree(data_));
            allocated_ = other.allocated_;
            size_ = other.size_;
            memory_ = other.memory_;
            memory_type_ = other.memory_type_;
            data_ = other.data_;
            other.allocated_ = false;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Reallocates memory for the vector.
     *
     * Frees existing memory if already allocated.
     *
     * @param size New number of elements.
     * @param memory_type Type of memory (global or managed).
     */
    void resize(const size_t size, const Memory_Type memory_type = Memory_Type::global) {
        size_ = size;
        memory_ = size_ * sizeof(T);
        memory_type_ = memory_type;
        if (allocated_) {
            CHECK(gpuFree(data_));
            allocated_ = false;
        }
        if (size_ == 0)
            return;
        if (memory_type_ == Memory_Type::global) {
            CHECK(gpuMalloc((void**)&data_, memory_));
            allocated_ = true;
        } else {
            CHECK(gpuMallocManaged((void**)&data_, memory_));
            allocated_ = true;
        }
    }

    /**
     * @brief Copies data from host memory to device memory.
     * @param h_data Pointer to host memory.
     */
    void copy_from_host(const T* h_data, cudaStream_t stream = nullptr) {
        if (size_ == 0)
            return;
        if (stream) {
            CHECK(cudaMemcpyAsync(data_, h_data, memory_, cudaMemcpyHostToDevice, stream));
        } else {
            CHECK(cudaMemcpy(data_, h_data, memory_, cudaMemcpyHostToDevice));
        }
    }

    /**
     * @brief Copies a specific number of elements from host memory to device memory.
     * @param h_data Pointer to host memory.
     * @param size Number of elements to copy.
     */
    void copy_from_host(const T* h_data, const size_t size, cudaStream_t stream = nullptr) {
        if (size == 0)
            return;
        const size_t memory = sizeof(T) * size;
        if (stream) {
            CHECK(cudaMemcpyAsync(data_, h_data, memory, cudaMemcpyHostToDevice, stream));
        } else {
            CHECK(cudaMemcpy(data_, h_data, memory, cudaMemcpyHostToDevice));
        }
    }

    /**
     * @brief Copies data from device memory to host memory.
     * @param h_data Pointer to host memory.
     */
    void copy_to_host(T* h_data, cudaStream_t stream = nullptr) const {
        if (size_ == 0)
            return;
        if (stream) {
            CHECK(cudaMemcpyAsync(h_data, data_, memory_, cudaMemcpyDeviceToHost, stream));
        } else {
            CHECK(cudaMemcpy(h_data, data_, memory_, cudaMemcpyDeviceToHost));
        }
    }

    /**
     * @brief Fills the entire vector with a constant value on the GPU.
     * @param value The value to fill with.
     */
    void fill(const T value, cudaStream_t stream = nullptr) {
        if (size_ == 0)
            return;
        if (memory_type_ == Memory_Type::global) {
            const int block_size = 256;
            const int grid_size = (size_ + block_size - 1) / block_size;
            gpu_fill<<<grid_size, block_size, 0, stream>>>(size_, value, data_);
            GPU_CHECK_KERNEL;
        } else  // managed (or unified) memory
        {
            if (stream)
                CHECK(cudaStreamSynchronize(stream));
            for (int i = 0; i < size_; ++i)
                data_[i] = value;
        }
    }

    /**
     * @brief Returns the number of elements in the vector.
     */
    size_t size() const { return size_; }

    /**
     * @brief Returns a pointer to the device data.
     */
    __host__ __device__ __forceinline__ T* data() { return data_; }

    /**
     * @brief Returns a const pointer to the device data.
     */
    __host__ __device__ __forceinline__ const T* data() const { return data_; }

  private:
    bool allocated_;          /**< True if memory is currently allocated */
    size_t size_;             /**< Number of elements */
    size_t memory_;           /**< Total memory in bytes */
    Memory_Type memory_type_; /**< Type of memory (global/managed) */
    T* data_;                 /**< Device memory pointer */
};
