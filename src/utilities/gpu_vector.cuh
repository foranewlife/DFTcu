#pragma once

#include "error.cuh"
#include "gpu_macro.cuh"

template <typename T>
static void __global__ gpu_fill(const size_t size, const T value, T* data) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        data[i] = value;
}

enum class Memory_Type {
    global = 0,  // global memory, also called (linear) device memory
    managed      // managed memory, also called unified memory
};

template <typename T>
class GPU_Vector {
  public:
    // default constructor
    GPU_Vector() {
        size_ = 0;
        memory_ = 0;
        memory_type_ = Memory_Type::global;
        allocated_ = false;
    }

    // only allocate memory
    GPU_Vector(const size_t size, const Memory_Type memory_type = Memory_Type::global) {
        allocated_ = false;
        resize(size, memory_type);
    }

    // deallocate memory
    ~GPU_Vector() {
        if (allocated_) {
            CHECK(gpuFree(data_));
            allocated_ = false;
        }
    }

    // Disable copy
    GPU_Vector(const GPU_Vector&) = delete;
    GPU_Vector& operator=(const GPU_Vector&) = delete;

    // Enable move
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

    // only allocate memory
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

    // copy data from host with the default size
    void copy_from_host(const T* h_data) {
        if (size_ == 0)
            return;
        CHECK(gpuMemcpy(data_, h_data, memory_, gpuMemcpyHostToDevice));
    }

    // copy data from host with a given size
    void copy_from_host(const T* h_data, const size_t size) {
        if (size == 0)
            return;
        const size_t memory = sizeof(T) * size;
        CHECK(gpuMemcpy(data_, h_data, memory, gpuMemcpyHostToDevice));
    }

    // copy data to host with the default size
    void copy_to_host(T* h_data) const {
        if (size_ == 0)
            return;
        CHECK(gpuMemcpy(h_data, data_, memory_, gpuMemcpyDeviceToHost));
    }

    // give "value" to each element
    void fill(const T value) {
        if (size_ == 0)
            return;
        if (memory_type_ == Memory_Type::global) {
            const int block_size = 128;
            const int grid_size = (size_ + block_size - 1) / block_size;
            gpu_fill<<<grid_size, block_size>>>(size_, value, data_);
            GPU_CHECK_KERNEL
        } else  // managed (or unified) memory
        {
            for (int i = 0; i < size_; ++i)
                data_[i] = value;
        }
    }

    // getters
    size_t size() const { return size_; }
    T const* data() const { return data_; }
    T* data() { return data_; }

  private:
    bool allocated_;
    size_t size_;
    size_t memory_;
    Memory_Type memory_type_;
    T* data_;
};
