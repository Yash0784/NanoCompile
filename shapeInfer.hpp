#ifndef SHAPEINFER_H
#define SHAPEINFER_H

#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include "onnx.pb.h"

// ============================================================================
// 1. Core Data Structures & Types
// ============================================================================
//
// NOTE ON THE REMOVED `Dim` TYPE:
// The original design represented every shape dimension as a `Dim`, a
// std::variant<int64_t, std::string, BinaryOp> so that a symbolic ONNX axis
// (e.g. a dynamic "batch_size" dim_param) could be carried through shape
// inference unresolved, and even combined algebraically (Dim + Dim).
//
// That machinery is removed here. Batch size is decided once, up front,
// before the graph is even parsed (see ONNXGraph::build_graph's new
// `batch_size` parameter). Every symbolic axis is substituted with a
// concrete integer at the single point where it enters the system --
// reading a ValueInfoProto's dim_param in getShape(). From then on every
// tensor's shape is a plain std::vector<int64_t>, so shape inference,
// broadcasting, and byte-size computation are all ordinary integer
// arithmetic -- no variant, no is_static()/get_static() checks, no
// unresolved placeholder strings.

enum class DataType : uint8_t{
    UNKNOWN = 0, 
    FLOAT32 = 1, 
    UINT8   = 2, 
    INT8    = 3, 
    INT32   = 6, 
    INT64   = 7, 
    FLOAT16 = 10
};

enum class Layout {
    NCHW,         // Vision Default (Channels First)
    NHWC,         // Vision Alternative (Channels Last)
    ROW_MAJOR,    // Standard Linear Algebra / Matrix 2D representation (C-style)
    COLUMN_MAJOR, // Contiguous Columns representation
    FLAT,         // 1D Vectors / Flattened Tensors 
    UNSPECIFIED   // Fallback layout for non-spatial operations
};

typedef struct TensorMetadata{
    uint32_t id;
    std::string name;

    DataType dtype;
    std::vector<int64_t> shape;   // now concrete ints only, no symbolic Dim
    Layout layout = Layout::UNSPECIFIED;

    const onnx::NodeProto* producer = nullptr;
    std::vector<const onnx::NodeProto*> consumers;

    bool is_initializer = false;
    bool is_constant = false;

    void* data = nullptr;

    size_t num_elements = 0;
    size_t bytes = 0;

    int buffer_id = -1;
    size_t offset = 0;

    int born_at = -1;
    int first_use = -1;
    int last_use = -1;

    bool shape_inferred = false;
    bool optimized = false;

    float swap_in_time = -1;
    float swap_out_time = -1;
}tmd;


void life_time_tens(const std::vector<onnx::NodeProto>& TopoNodes, std::unordered_map<std::string, tmd*>& master_tensor_map);

// ============================================================================
// 2. Byte-size computation
// ============================================================================

// Number of bytes a single element of `dtype` occupies.
size_t dtype_size_bytes(DataType dtype);

// Given a tensor whose `shape` is already fully concrete (every axis
// resolved, batch included), compute and fill in `num_elements` and
// `bytes`. Call this at the exact moment a tensor's shape is finalized --
// right after reading an initializer's shape, right after resolving a
// ValueInfoProto's shape, or right after an infer_* function has written
// outputs[i]->shape. It does not know about batch size or anything
// symbolic; it assumes the shape is already a plain vector<int64_t>.
void compute_tensor_bytes(tmd* tensor);

// ============================================================================
// 3. Core Shape Inference Functions (Your 10 Operators)
// ============================================================================

// 1. Element-wise Activation
void infer_relu(const onnx::NodeProto& node, 
                const std::vector<tmd*>& inputs, 
                std::vector<tmd*>& outputs);

// 2. Element-wise Broadcast Addition
void infer_add(const onnx::NodeProto& node, 
               const std::vector<tmd*>& inputs, 
               std::vector<tmd*>& outputs);

// 3. Standard / Batch Matrix Multiplication
void infer_matmul(const onnx::NodeProto& node, 
                  const std::vector<tmd*>& inputs, 
                  std::vector<tmd*>& outputs);

// 4. General Matrix Multiplication (2D with optional transpositions)
void infer_gemm(const onnx::NodeProto& node, 
                 const std::vector<tmd*>& inputs, 
                 std::vector<tmd*>& outputs);

// 5. N-Dimensional / 2D Convolution spatial downsampling
void infer_conv(const onnx::NodeProto& node, 
                 const std::vector<tmd*>& inputs, 
                 std::vector<tmd*>& outputs);

// 6 & 7. MaxPool and AveragePool Spatial Reduction
void infer_pooling(const onnx::NodeProto& node, 
                   const std::vector<tmd*>& inputs, 
                   std::vector<tmd*>& outputs);

// 8. Structural Layout Alteration
void infer_reshape(const onnx::NodeProto& node, 
                   const std::vector<tmd*>& inputs, 
                   std::vector<tmd*>& outputs, 
                   const google::protobuf::RepeatedPtrField<onnx::TensorProto>& initializers);

// 9. Axis Permutation / Swapping
void infer_transpose(const onnx::NodeProto& node, 
                     const std::vector<tmd*>& inputs, 
                     std::vector<tmd*>& outputs);

// 10. Multi-tensor Axis Concatenation
void infer_concat(const onnx::NodeProto& node, 
                  const std::vector<tmd*>& inputs, 
                  std::vector<tmd*>& outputs);

// ============================================================================
// 4. Internal Engine Helpers
// ============================================================================

// Right-aligned broadcast rule for a single pair of dims: whichever side is
// 1 defers to the other; if neither is 1 the exported graph guarantees they
// match, so either can be returned. Plain int64_t now -- no Dim wrapper.
int64_t broadcast_dims_optimistic(int64_t a, int64_t b);

// Protobuf extraction helpers
int64_t get_node_attr_int(const onnx::NodeProto& node, const std::string& name, int64_t default_val);
std::vector<int64_t> get_node_attr_ints(const onnx::NodeProto& node, const std::string& name);

// Helpers to print.
const char* data_type_to_string(DataType dtype);
const char* layout_to_string(Layout layout);
void infershape_driver(const onnx::NodeProto& node, std::string op, const google::protobuf::RepeatedPtrField<onnx::TensorProto>& initializers, std::vector<tmd*>& node_inputs, std::vector<tmd*>& node_outputs);

#endif // SHAPEINFER_H
