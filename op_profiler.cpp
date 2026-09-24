#include <iostream>
#include <string>
#include <vector>
#include <numeric>   // std::accumulate
#include <algorithm>
#include <utility>   // std::pair, std::tie
#include <cstring>   // std::memset

#include "shapeInfer.hpp"
#include "op_profiler.hpp" // declares profile_node -- included here so the
                           // compiler checks this file's definition against
                           // that declaration (catches signature drift
                           // immediately, not at link time).

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
// NOTE: <cudnn_graph.h> (the newer cuDNN "graph" front-end API) was in the
// original draft but isn't needed here -- every op below uses the classic
// cuDNN API (cudnnConvolutionForward, cudnnActivationForward, etc.), which
// lives in plain <cudnn.h>.

// ============================================================================
// 1. TIMING HELPERS  (your getTime/profile, fixed)
// ============================================================================

template<typename Fn>
float getTime(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop, Fn work){
    CUDA_CHECK(cudaEventRecord(start, stream));
    work();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

// Fixed: accumulate() needs the std:: prefix and a starting value (0.0f),
// otherwise it doesn't resolve/compile. Kept as a mean here (matching your
// draft) -- median is a bit more outlier-resistant if you want to swap it
// in later (sort `times` and take the middle element instead).
template<typename Fn>
float profile(cudaStream_t stream, int warmup, int iters, Fn work){
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for(int i = 0; i < warmup; i++){
        getTime(stream, start, stop, work);
    }

    std::vector<float> times;
    times.reserve(iters);
    for(int i = 0; i < iters; i++){
        times.push_back(getTime(stream, start, stop, work));
    }

    float total = std::accumulate(times.begin(), times.end(), 0.0f);
    float avg_time = total / (float)times.size();

    // Fixed: cudaEventDestroy takes the event by VALUE, not a pointer to it
    // (cudaEventCreate took a pointer because it needed to WRITE the handle
    // back to you; destroy just needs the handle itself).
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return avg_time;
}


// ============================================================================
// 2. GENERIC DEVICE BUFFER HELPERS
// ============================================================================
// One allocator that works for ANY tmd*, for any op type, because it only
// ever needs ->bytes -- which your parser already computed for every
// tensor in the graph, regardless of which model produced it.

struct DeviceBuffer {
    tmd* tensor = nullptr;
    void* d_ptr = nullptr;
};

DeviceBuffer alloc_device_buffer(tmd* t, cudaStream_t stream){
    DeviceBuffer buf;
    buf.tensor = t;
    CUDA_CHECK(cudaMalloc(&buf.d_ptr, t->bytes));
    // Values are irrelevant for timing -- just need valid, sized memory.
    CUDA_CHECK(cudaMemsetAsync(buf.d_ptr, 0, t->bytes, stream));
    return buf;
}

std::vector<DeviceBuffer> alloc_all(const std::vector<tmd*>& tensors, cudaStream_t stream){
    std::vector<DeviceBuffer> bufs;
    bufs.reserve(tensors.size());
    for (tmd* t : tensors) bufs.push_back(alloc_device_buffer(t, stream));
    return bufs;
}

void free_buffer(DeviceBuffer& buf){
    if (buf.d_ptr) CUDA_CHECK(cudaFree(buf.d_ptr));
    buf.d_ptr = nullptr;
}

void free_all(std::vector<DeviceBuffer>& bufs){
    for (auto& b : bufs) free_buffer(b);
}

// cuDNN needs its own dtype enum -- separate from your DataType, same
// pattern as dtype_size_bytes() in shapeInfer.cpp but for cuDNN's world.
cudnnDataType_t onnx_dtype_to_cudnn(DataType dtype){
    switch (dtype) {
        case DataType::FLOAT32: return CUDNN_DATA_FLOAT;
        case DataType::FLOAT16: return CUDNN_DATA_HALF;
        case DataType::INT8:    return CUDNN_DATA_INT8;
        case DataType::INT32:   return CUDNN_DATA_INT32;
        default:
            std::cerr << "[-] Error: no cuDNN dtype mapping for "
                      << data_type_to_string(dtype) << "\n";
            std::exit(1);
    }
}

// Elementwise ops (Relu, elementwise Add) don't care about a tensor's real
// rank -- only its total element count. Describing every tensor as a flat
// [1, num_elements, 1, 1] NCHW tensor lets one helper serve 2D MLP
// activations and 4D conv activations identically.
cudnnTensorDescriptor_t make_flat_descriptor(tmd* t){
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        desc, CUDNN_TENSOR_NCHW, onnx_dtype_to_cudnn(t->dtype),
        1, (int)t->num_elements, 1, 1));
    return desc;
}


// ============================================================================
// 3. PER-OP EXECUTORS
// ============================================================================
// Each one mirrors its infer_* counterpart in shapeInfer.cpp: same node,
// same attribute reads, same input tensors. The difference is these read
// outputs[0]->shape/bytes directly (already computed by shape inference
// during build_graph) instead of re-deriving the output shape themselves --
// no need to duplicate infer_gemm's/infer_conv's math here.

double profile_relu(const onnx::NodeProto& /*node*/,
                     const std::vector<tmd*>& inputs,
                     const std::vector<tmd*>& outputs,
                     cudnnHandle_t cudnn_h, cudaStream_t stream){
    auto in_buf  = alloc_device_buffer(inputs[0], stream);
    auto out_buf = alloc_device_buffer(outputs[0], stream);

    cudnnTensorDescriptor_t desc = make_flat_descriptor(inputs[0]);

    cudnnActivationDescriptor_t act;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act));
    CUDNN_CHECK(cudnnSetActivationDescriptor(
        act, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, /*coef=*/0.0));

    float alpha = 1.0f, beta = 0.0f;
    double ms = profile(stream, 5, 20, [&](){
        CUDNN_CHECK(cudnnActivationForward(
            cudnn_h, act, &alpha, desc, in_buf.d_ptr, &beta, desc, out_buf.d_ptr));
    });

    CUDNN_CHECK(cudnnDestroyActivationDescriptor(act));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    free_buffer(in_buf);
    free_buffer(out_buf);
    return ms;
}

double profile_add(const onnx::NodeProto& /*node*/,
                    const std::vector<tmd*>& inputs,
                    const std::vector<tmd*>& outputs,
                    cudnnHandle_t cudnn_h, cudaStream_t stream){
    auto bufA   = alloc_device_buffer(inputs[0], stream);
    auto bufB   = alloc_device_buffer(inputs[1], stream);
    auto bufOut = alloc_device_buffer(outputs[0], stream);

    // Two cases, matching infer_add's broadcasting logic: same element
    // count on both sides (pure elementwise) vs. a smaller bias tensor
    // being broadcast in (e.g. Gemm-style bias-add).
    bool same_shape = (inputs[0]->num_elements == inputs[1]->num_elements);

    double ms;
    if (same_shape) {
        cudnnTensorDescriptor_t descA = make_flat_descriptor(inputs[0]);

        cudnnOpTensorDescriptor_t opDesc;
        CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&opDesc));
        CUDNN_CHECK(cudnnSetOpTensorDescriptor(
            opDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN));

        float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
        ms = profile(stream, 5, 20, [&](){
            CUDNN_CHECK(cudnnOpTensor(
                cudnn_h, opDesc,
                &alpha1, descA, bufA.d_ptr,
                &alpha2, descA, bufB.d_ptr,
                &beta,   descA, bufOut.d_ptr));
        });

        CUDNN_CHECK(cudnnDestroyOpTensorDescriptor(opDesc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(descA));
    } else {
        // Broadcast case: cudnnAddTensor computes C = alpha*B + beta*C, so
        // first copy A into the output buffer (as the starting C), then
        // accumulate the smaller B (e.g. a bias vector) into it with beta=1.
        cudnnTensorDescriptor_t descOut = make_flat_descriptor(outputs[0]);
        cudnnTensorDescriptor_t descB   = make_flat_descriptor(inputs[1]);

        float alpha = 1.0f, beta_keep = 1.0f;
        ms = profile(stream, 5, 20, [&](){
            CUDA_CHECK(cudaMemcpyAsync(bufOut.d_ptr, bufA.d_ptr, outputs[0]->bytes,
                                        cudaMemcpyDeviceToDevice, stream));
            CUDNN_CHECK(cudnnAddTensor(
                cudnn_h, &alpha, descB, bufB.d_ptr, &beta_keep, descOut, bufOut.d_ptr));
        });

        CUDNN_CHECK(cudnnDestroyTensorDescriptor(descOut));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(descB));
    }

    free_buffer(bufA);
    free_buffer(bufB);
    free_buffer(bufOut);
    return ms;
}

// Shared by MatMul (no transpose attributes) and Gemm (has transA/transB).
// Mirrors infer_gemm's/infer_matmul's shape reads, but sizes the output
// buffer from outputs[0]->bytes directly rather than recomputing M*N.
double profile_gemm_like(const onnx::NodeProto& node, bool has_trans_attrs,
                          const std::vector<tmd*>& inputs,
                          const std::vector<tmd*>& outputs,
                          cublasHandle_t cublas_h, cudaStream_t stream){
    int64_t transA = has_trans_attrs ? get_node_attr_int(node, "transA", 0) : 0;
    int64_t transB = has_trans_attrs ? get_node_attr_int(node, "transB", 0) : 0;

    const auto& shape_A = inputs[0]->shape;
    const auto& shape_B = inputs[1]->shape;
    int64_t M = (transA == 0) ? shape_A[0] : shape_A[1];
    int64_t K = (transA == 0) ? shape_A[1] : shape_A[0];
    int64_t N = (transB == 0) ? shape_B[1] : shape_B[0];

    auto bufA   = alloc_device_buffer(inputs[0], stream);
    auto bufB   = alloc_device_buffer(inputs[1], stream);
    auto bufOut = alloc_device_buffer(outputs[0], stream); // sized via outputs[0]->bytes

    float alpha = 1.0f, beta = 0.0f;
    cublasOperation_t opA = (transA == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB = (transB == 0) ? CUBLAS_OP_N : CUBLAS_OP_T;

    // Row-major -> column-major trick (see gpu_profiling_example.cpp for
    // the full explanation): swap A/B and their transpose flags, pass
    // dims as (N, M, K) instead of (M, N, K).
    double ms = profile(stream, 5, 20, [&](){
        CUBLAS_CHECK(cublasSgemm(
            cublas_h, opB, opA, (int)N, (int)M, (int)K,
            &alpha,
            (float*)bufB.d_ptr, (int)((transB == 0) ? N : K),
            (float*)bufA.d_ptr, (int)((transA == 0) ? K : M),
            &beta,
            (float*)bufOut.d_ptr, (int)N));
    });

    free_buffer(bufA);
    free_buffer(bufB);
    free_buffer(bufOut);
    return ms;
}

double profile_gemm(const onnx::NodeProto& node,
                     const std::vector<tmd*>& inputs,
                     const std::vector<tmd*>& outputs,
                     cublasHandle_t cublas_h, cudaStream_t stream){
    // NOTE: this ignores Gemm's optional bias input C for timing purposes --
    // cublasSgemm itself has no bias argument. If you want the bias-add
    // folded into this measurement, chain a profile_add() call using
    // outputs[0] and inputs[2] after this, and sum the two times.
    return profile_gemm_like(node, /*has_trans_attrs=*/true, inputs, outputs, cublas_h, stream);
}

double profile_matmul(const onnx::NodeProto& node,
                       const std::vector<tmd*>& inputs,
                       const std::vector<tmd*>& outputs,
                       cublasHandle_t cublas_h, cudaStream_t stream){
    if (inputs[0]->shape.size() != 2 || inputs[1]->shape.size() != 2) {
        std::cerr << "[-] Warning: profile_matmul only handles 2D operands "
                     "for now (batched MatMul not implemented). Returning 0.\n";
        return 0.0;
    }
    return profile_gemm_like(node, /*has_trans_attrs=*/false, inputs, outputs, cublas_h, stream);
}

double profile_conv(const onnx::NodeProto& node,
                     const std::vector<tmd*>& inputs,
                     const std::vector<tmd*>& outputs,
                     cudnnHandle_t cudnn_h, cudaStream_t stream){
    const auto& input_shape  = inputs[0]->shape;  // [N, C, H, W]
    const auto& weight_shape = inputs[1]->shape;  // [K, C, R, S]

    if (input_shape.size() != 4 || weight_shape.size() != 4) {
        std::cerr << "[-] Warning: profile_conv only handles 2D (4D-shaped) "
                     "convolutions for now. Returning 0.\n";
        return 0.0;
    }

    std::vector<int64_t> strides = get_node_attr_ints(node, "strides");
    if (strides.empty()) strides = {1, 1};
    std::vector<int64_t> dilations = get_node_attr_ints(node, "dilations");
    if (dilations.empty()) dilations = {1, 1};
    std::vector<int64_t> pads = get_node_attr_ints(node, "pads");
    if (pads.empty()) pads = {0, 0, 0, 0};
    // KNOWN LIMITATION: auto_pad ("SAME_UPPER"/"SAME_LOWER") isn't
    // translated into explicit cuDNN padding here -- infer_conv already
    // computed the correct OUTPUT shape for that case (used below via
    // outputs[0]->shape), but the padding cuDNN actually applies during
    // the timed call may not exactly match "SAME" behavior. For pure
    // timing purposes this is usually close enough (padding amount barely
    // affects runtime), but flag it if you need bit-exact behavior later.

    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnFilterDescriptor_t filt_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, onnx_dtype_to_cudnn(inputs[0]->dtype),
        (int)input_shape[0], (int)input_shape[1], (int)input_shape[2], (int)input_shape[3]));

    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filt_desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filt_desc, onnx_dtype_to_cudnn(inputs[1]->dtype), CUDNN_TENSOR_NCHW,
        (int)weight_shape[0], (int)weight_shape[1], (int)weight_shape[2], (int)weight_shape[3]));

    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc, (int)pads[0], (int)pads[1], (int)strides[0], (int)strides[1],
        (int)dilations[0], (int)dilations[1], CUDNN_CROSS_CORRELATION,
        onnx_dtype_to_cudnn(inputs[0]->dtype)));

    // Output shape comes straight from shape inference -- no need to
    // recompute it here.
    const auto& out_shape = outputs[0]->shape;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, onnx_dtype_to_cudnn(outputs[0]->dtype),
        (int)out_shape[0], (int)out_shape[1], (int)out_shape[2], (int)out_shape[3]));

    // Let cuDNN pick a reasonable algorithm automatically -- good enough
    // for profiling; a real runtime might benchmark several and cache the
    // fastest per shape instead.
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_h, in_desc, filt_desc, conv_desc, out_desc, algo, &workspace_bytes));

    void* d_workspace = nullptr;
    if (workspace_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));
    }

    auto bufIn  = alloc_device_buffer(inputs[0], stream);
    auto bufW   = alloc_device_buffer(inputs[1], stream);
    auto bufOut = alloc_device_buffer(outputs[0], stream);

    float alpha = 1.0f, beta = 0.0f;
    double ms = profile(stream, 5, 20, [&](){
        CUDNN_CHECK(cudnnConvolutionForward(
            cudnn_h, &alpha, in_desc, bufIn.d_ptr, filt_desc, bufW.d_ptr,
            conv_desc, algo, d_workspace, workspace_bytes,
            &beta, out_desc, bufOut.d_ptr));
    });

    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
    free_buffer(bufIn);
    free_buffer(bufW);
    free_buffer(bufOut);
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filt_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    return ms;
}

double profile_pooling(const onnx::NodeProto& node, const std::string& op,
                        const std::vector<tmd*>& inputs,
                        const std::vector<tmd*>& outputs,
                        cudnnHandle_t cudnn_h, cudaStream_t stream){
    const auto& input_shape = inputs[0]->shape; // [N, C, H, W]
    if (input_shape.size() != 4) {
        std::cerr << "[-] Warning: profile_pooling only handles 4D (2D spatial) "
                     "pooling for now. Returning 0.\n";
        return 0.0;
    }

    std::vector<int64_t> kernel_shape = get_node_attr_ints(node, "kernel_shape");
    std::vector<int64_t> strides = get_node_attr_ints(node, "strides");
    if (strides.empty()) strides = {1, 1};
    std::vector<int64_t> pads = get_node_attr_ints(node, "pads");
    if (pads.empty()) pads = {0, 0, 0, 0};

    cudnnPoolingMode_t mode = (op == "MaxPool")
        ? CUDNN_POOLING_MAX
        : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

    cudnnPoolingDescriptor_t pool_desc;
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        pool_desc, mode, CUDNN_PROPAGATE_NAN,
        (int)kernel_shape[0], (int)kernel_shape[1],
        (int)pads[0], (int)pads[1],
        (int)strides[0], (int)strides[1]));

    cudnnTensorDescriptor_t in_desc, out_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, onnx_dtype_to_cudnn(inputs[0]->dtype),
        (int)input_shape[0], (int)input_shape[1], (int)input_shape[2], (int)input_shape[3]));

    const auto& out_shape = outputs[0]->shape; // already computed by infer_pooling
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, onnx_dtype_to_cudnn(outputs[0]->dtype),
        (int)out_shape[0], (int)out_shape[1], (int)out_shape[2], (int)out_shape[3]));

    auto bufIn  = alloc_device_buffer(inputs[0], stream);
    auto bufOut = alloc_device_buffer(outputs[0], stream);

    float alpha = 1.0f, beta = 0.0f;
    double ms = profile(stream, 5, 20, [&](){
        CUDNN_CHECK(cudnnPoolingForward(
            cudnn_h, pool_desc, &alpha, in_desc, bufIn.d_ptr, &beta, out_desc, bufOut.d_ptr));
    });

    free_buffer(bufIn);
    free_buffer(bufOut);
    CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pool_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(out_desc));
    return ms;
}

// Reshape/Transpose/Concat have no meaningful GPU compute cost to profile
// in the general case (Reshape is a metadata-only reinterpretation of the
// same memory; Concat of already-contiguous inputs is effectively a
// memcpy; Transpose only costs real time if it changes physical layout,
// which most exported graphs avoid). Stubbed at 0 rather than spending
// profiling effort here -- revisit only if your swap-plan simulator shows
// these nodes actually sitting on the critical path for some model.
double profile_zero_cost(const onnx::NodeProto& /*node*/){
    return 0.0;
}


// ============================================================================
// 4. DISPATCHER  (mirrors infershape_driver's op -> function mapping)
// ============================================================================

double profile_node(const onnx::NodeProto& node, const std::string& op,
                     const std::vector<tmd*>& node_inputs,
                     const std::vector<tmd*>& node_outputs,
                     cublasHandle_t cublas_h, cudnnHandle_t cudnn_h,
                     cudaStream_t stream){
    if (op == "Relu") {
        return profile_relu(node, node_inputs, node_outputs, cudnn_h, stream);
    }
    else if (op == "Add") {
        return profile_add(node, node_inputs, node_outputs, cudnn_h, stream);
    }
    else if (op == "MatMul") {
        return profile_matmul(node, node_inputs, node_outputs, cublas_h, stream);
    }
    else if (op == "Gemm") {
        return profile_gemm(node, node_inputs, node_outputs, cublas_h, stream);
    }
    else if (op == "Conv") {
        return profile_conv(node, node_inputs, node_outputs, cudnn_h, stream);
    }
    else if (op == "MaxPool" || op == "AveragePool") {
        return profile_pooling(node, op, node_inputs, node_outputs, cudnn_h, stream);
    }
    else if (op == "Reshape" || op == "Transpose" || op == "Concat") {
        return profile_zero_cost(node);
    }
    else {
        std::cerr << "[-] Error: no profiler implemented for op '" << op
                  << "' on node '" << node.name() << "'. Recording 0.\n";
        return 0.0;
    }
}

// ============================================================================
// 5. PCIe BANDWIDTH CALIBRATION + PER-TENSOR SWAP TIMES
// ============================================================================
// Real PCIe copy time isn't just bytes/bandwidth -- every cudaMemcpyAsync
// call has a roughly-fixed per-call overhead (driver call, DMA setup) that
// dominates at small sizes and is negligible at large ones. So instead of
// trusting one flat GB/s number, we sweep real transfers across a range of
// sizes and least-squares fit an affine model per direction:
//
//     time_ms(bytes) = latency_ms + slope_ms_per_byte * bytes
//
// then apply that model to every tensor's actual ->bytes to fill in
// ->swap_in_time / ->swap_out_time. Same warmup+average discipline as
// profile_node above, just timing memcpys instead of cuDNN/cuBLAS calls.

double PCIeBandwidthModel::h2d_bandwidth_GBps() const {
    if (h2d_slope_ms_per_byte <= 0.0) return 0.0;
    double bytes_per_ms = 1.0 / h2d_slope_ms_per_byte;
    return bytes_per_ms * 1000.0 / 1e9; // bytes/ms -> bytes/s -> GB/s
}
double PCIeBandwidthModel::d2h_bandwidth_GBps() const {
    if (d2h_slope_ms_per_byte <= 0.0) return 0.0;
    double bytes_per_ms = 1.0 / d2h_slope_ms_per_byte;
    return bytes_per_ms * 1000.0 / 1e9;
}

namespace {

// Least-squares fit of ms = a + b*bytes over the collected (bytes, ms)
// samples. Returns {intercept_ms, slope_ms_per_byte}.
std::pair<double, double> fit_affine(const std::vector<double>& xs_bytes,
                                      const std::vector<double>& ys_ms) {
    size_t n = xs_bytes.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (size_t i = 0; i < n; ++i) {
        sum_x  += xs_bytes[i];
        sum_y  += ys_ms[i];
        sum_xy += xs_bytes[i] * ys_ms[i];
        sum_xx += xs_bytes[i] * xs_bytes[i];
    }
    double denom = (double)n * sum_xx - sum_x * sum_x;
    double b = (denom != 0.0) ? ((double)n * sum_xy - sum_x * sum_y) / denom : 0.0;
    double a = (sum_y - b * sum_x) / (double)n;
    // Both latency and slope should physically be >= 0. Noise at the
    // smallest sizes can pull the fit line slightly negative at the
    // intercept -- clamp so small tensors never get a nonsensical
    // negative swap time.
    if (a < 0.0) a = 0.0;
    if (b < 0.0) b = 0.0;
    return {a, b};
}

// Geometric sweep from 4KB up to max_bytes (4x growth per step -- keeps
// the sweep short while still covering the fixed-overhead regime at the
// small end and the bandwidth-bound regime at the large end).
std::vector<size_t> make_sweep_sizes(size_t max_bytes) {
    std::vector<size_t> sizes;
    size_t s = 4ull * 1024; // 4 KB floor
    while (s < max_bytes) {
        sizes.push_back(s);
        s *= 4;
    }
    sizes.push_back(max_bytes); // always include the actual allocation ceiling
    return sizes;
}

} // namespace

PCIeBandwidthModel profile_pcie_bandwidth(cudaStream_t h2d_stream,
                                           cudaStream_t d2h_stream){
    // Measure real free memory rather than trusting a nominal number --
    // same principle as budgeting the 6GB against actual runtime
    // free-memory (CUDA context / cuDNN workspace already eat some of it).
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    // Need a pinned host buffer AND a device buffer of this size at the
    // same time, plus headroom for whatever runs after calibration --
    // stay well under what's actually free.
    size_t max_bytes = free_bytes / 4;
    const size_t hard_cap = 512ull * 1024 * 1024; // bandwidth-bound regime is already visible well before this
    if (max_bytes > hard_cap) max_bytes = hard_cap;
    if (max_bytes < 1ull * 1024 * 1024) {
        std::cerr << "[!] Warning: only " << (free_bytes / 1024.0 / 1024.0)
                  << " MB free for PCIe calibration -- results may be unreliable.\n";
        if (max_bytes < 64ull * 1024) max_bytes = 64ull * 1024; // absolute floor so the sweep can still run
    }

    std::vector<size_t> sizes = make_sweep_sizes(max_bytes);

    // One pinned host buffer + one device buffer, both sized to the sweep
    // ceiling; each step just transfers a prefix of them. Pinned matters:
    // pageable host memory would force the driver through an internal
    // bounce buffer, measuring a slower number that doesn't match what
    // the real swap-in/out copies (also pinned) will actually see.
    void* h_ptr = nullptr;
    void* d_ptr = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_ptr, max_bytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaMalloc(&d_ptr, max_bytes));
    std::memset(h_ptr, 0, max_bytes);
    CUDA_CHECK(cudaMemset(d_ptr, 0, max_bytes));

    std::vector<double> h2d_bytes, h2d_ms, d2h_bytes, d2h_ms;
    const int warmup = 3, iters = 10;

    for (size_t sz : sizes) {
        double ms_h2d = profile(h2d_stream, warmup, iters, [&](){
            CUDA_CHECK(cudaMemcpyAsync(d_ptr, h_ptr, sz, cudaMemcpyHostToDevice, h2d_stream));
        });
        h2d_bytes.push_back((double)sz);
        h2d_ms.push_back(ms_h2d);

        double ms_d2h = profile(d2h_stream, warmup, iters, [&](){
            CUDA_CHECK(cudaMemcpyAsync(h_ptr, d_ptr, sz, cudaMemcpyDeviceToHost, d2h_stream));
        });
        d2h_bytes.push_back((double)sz);
        d2h_ms.push_back(ms_d2h);
    }

    CUDA_CHECK(cudaFreeHost(h_ptr));
    CUDA_CHECK(cudaFree(d_ptr));

    PCIeBandwidthModel model;
    std::tie(model.h2d_latency_ms, model.h2d_slope_ms_per_byte) = fit_affine(h2d_bytes, h2d_ms);
    std::tie(model.d2h_latency_ms, model.d2h_slope_ms_per_byte) = fit_affine(d2h_bytes, d2h_ms);

    std::cout << "[PCIe calibration] swept " << sizes.size() << " sizes up to "
              << (max_bytes / 1024.0 / 1024.0) << " MB\n"
              << "  H2D: latency=" << model.h2d_latency_ms << " ms, bandwidth="
              << model.h2d_bandwidth_GBps() << " GB/s\n"
              << "  D2H: latency=" << model.d2h_latency_ms << " ms, bandwidth="
              << model.d2h_bandwidth_GBps() << " GB/s\n";

    return model;
}

void compute_tensor_swap_times(const std::vector<tmd*>& tensors,
                                const PCIeBandwidthModel& model){
    for (tmd* t : tensors) {
        if (!t || t->bytes == 0) continue;
        double bytes = (double)t->bytes;
        t->swap_in_time  = (float)(model.h2d_latency_ms + model.h2d_slope_ms_per_byte * bytes);
        t->swap_out_time = (float)(model.d2h_latency_ms + model.d2h_slope_ms_per_byte * bytes);
    }
}