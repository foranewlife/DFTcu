#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cusolverDn.h>

// memory manipulation
#define gpuMalloc cudaMalloc
#define gpuMallocManaged cudaMallocManaged
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyFromSymbol cudaMemcpyFromSymbol
#define gpuMemcpyToSymbol cudaMemcpyToSymbol
#define gpuGetSymbolAddress cudaGetSymbolAddress
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToHost cudaMemcpyHostToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemset cudaMemset

// error handling
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError

// device manipulation
#define gpuSetDevice cudaSetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuDeviceProp cudaDeviceProp
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceCanAccessPeer cudaDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
#define gpuDeviceSynchronize cudaDeviceSynchronize

// stream
#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy

// random numbers
#define gpurandState curandState
#define gpurand_normal_double curand_normal_double
#define gpurand_normal curand_normal
#define gpurand_init curand_init

// blas
#define gpublasHandle_t cublasHandle_t
#define gpublasSgemv cublasSgemv
#define gpublasSgemm cublasSgemm
#define gpublasSdgmm cublasSdgmm
#define gpublasDgemv cublasDgemv
#define gpublasDestroy cublasDestroy
#define gpublasCreate cublasCreate
#define GPUBLAS_SIDE_LEFT CUBLAS_SIDE_LEFT
#define GPUBLAS_OP_N CUBLAS_OP_N
#define GPUBLAS_OP_T CUBLAS_OP_T

// lapack
#define gpuDoubleComplex cuDoubleComplex
#define gpusolverDnHandle_t cusolverDnHandle_t
#define gpusolverDnCreate cusolverDnCreate
#define gpusolverDnDestroy cusolverDnDestroy
#define gpusolverEigMode_t cusolverEigMode_t
#define gpusolverFillMode_t cublasFillMode_t
#define GPUSOLVER_EIG_MODE_NOVECTOR CUSOLVER_EIG_MODE_NOVECTOR
#define GPUSOLVER_EIG_MODE_VECTOR CUSOLVER_EIG_MODE_VECTOR
#define GPUSOLVER_FILL_MODE_LOWER CUBLAS_FILL_MODE_LOWER
#define gpusolverSyevjInfo_t syevjInfo_t
#define gpusolverDnCreateSyevjInfo cusolverDnCreateSyevjInfo
#define gpusolverDnDestroySyevjInfo cusolverDnDestroySyevjInfo
#define gpusolverDnZheevj_bufferSize cusolverDnZheevj_bufferSize
#define gpusolverDnZheevj cusolverDnZheevj
#define gpusolverDnZheevd_bufferSize cusolverDnZheevd_bufferSize
#define gpusolverDnZheevd cusolverDnZheevd
#define gpusolverDnDsyevj_bufferSize cusolverDnDsyevj_bufferSize
#define gpusolverDnDsyevj cusolverDnDsyevj
#define gpusolverDnZheevjBatched_bufferSize cusolverDnZheevjBatched_bufferSize
#define gpusolverDnZheevjBatched cusolverDnZheevjBatched

// FFT
#define gpufftHandle cufftHandle
#define gpufftComplex cufftDoubleComplex
#define gpufftExecZ2Z cufftExecZ2Z
#define gpufftPlan3d cufftPlan3d
#define gpufftPlanMany cufftPlanMany
#define gpufftDestroy cufftDestroy
#define GPUFFT_SUCCESS CUFFT_SUCCESS
#define GPUFFT_Z2Z CUFFT_Z2Z
#define GPUFFT_FORWARD CUFFT_FORWARD
#define GPUFFT_INVERSE CUFFT_INVERSE
