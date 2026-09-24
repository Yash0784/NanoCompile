#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>

#include "onnx.pb.h"
#include "shapeInfer.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#define CUDA_CHECK(exp) do {                                               \
    cudaError_t err__ = (exp);                                             \
    if (err__ != cudaSuccess) {                                            \
        std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err__)          \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
        std::exit(1);                                                     \
    }                                                                      \
} while (0)

#define CUDNN_CHECK(exp) do {                                              \
    cudnnStatus_t st__ = (exp);                                            \
    if (st__ != CUDNN_STATUS_SUCCESS) {                                    \
        std::cerr << "[CUDNN ERROR] " << cudnnGetErrorString(st__)         \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
        std::exit(1);                                                     \
    }                                                                      \
} while (0)

#define CUBLAS_CHECK(exp) do {                                             \
    cublasStatus_t st__ = (exp);                                           \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
        std::cerr << "[CUBLAS ERROR] status=" << st__                      \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
        std::exit(1);                                                     \
    }                                                                      \
} while (0)

// Data structure holding calibrated PCIe bandwidth and transfer latency parameters
struct PCIeBandwidthModel {
    double h2d_latency_ms = 0.0;
    double h2d_slope_ms_per_byte = 0.0;
    double d2h_latency_ms = 0.0;
    double d2h_slope_ms_per_byte = 0.0;

    double h2d_bandwidth_GBps() const;
    double d2h_bandwidth_GBps() const;
};

// Profiles a single node's compute time on the GPU by allocating dummy
// device buffers sized from each tensor's ->bytes, dispatching to the
// correct cuBLAS/cuDNN executor for `op`, and timing it with CUDA events
// (warmup + averaged iterations). Returns the measured time in
// milliseconds. Ops with no implemented executor (or an unsupported
// shape, e.g. batched MatMul) print a warning and return 0.0 rather than
// aborting the whole run.
double profile_node(const onnx::NodeProto& node, const std::string& op,
                     const std::vector<tmd*>& node_inputs,
                     const std::vector<tmd*>& node_outputs,
                     cublasHandle_t cublas_h, cudnnHandle_t cudnn_h,
                     cudaStream_t stream);

// Measures PCIe Host-to-Device and Device-to-Host transfer latencies and slopes
// across a range of payload sizes to create an affine timing model.
PCIeBandwidthModel profile_pcie_bandwidth(cudaStream_t h2d_stream,
                                           cudaStream_t d2h_stream);

// Uses the PCIe bandwidth model to calculate and populate the swap_in_time and
// swap_out_time fields for each tensor metadata struct in the graph.
void compute_tensor_swap_times(const std::vector<tmd*>& tensors,
                                const PCIeBandwidthModel& model);