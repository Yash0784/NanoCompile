#ifndef SHAPEINFER_H
#define SHAPEINFER_H

#include <vector>
#include <string>
#include <variant>
#include <cstdint>
#include "onnx.pb.h"


// ============================================================================
// 1. Core Data Structures & Types
// ============================================================================

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

enum class OpType { ADD, SUB, MUL, DIV };

struct Dim{
    struct BinaryOp{ 
        OpType op; 
        Dim* left; 
        Dim* right; 
    };

    std::variant<int64_t, std::string, BinaryOp> expr;

    Dim(int64_t val);
    Dim(std::string var);
    Dim(OpType op, Dim* l, Dim* r);

    bool is_static() const;
    int64_t get_static() const;
};

typedef struct TensorMetadata{
    uint32_t id;
    std::string name;

    DataType dtype;
    std::vector<Dim> shape;
    Layout layout = Layout::UNSPECIFIED;

    onnx::NodeProto producer;
    std::vector<onnx::NodeProto> consumers;

    bool is_initializer = false;
    bool is_constant = false;

    void* data = nullptr;

    size_t num_elements;
    size_t bytes;

    int buffer_id = -1;
    size_t offset = 0;

    int first_use = -1;
    int last_use = -1;

    bool shape_inferred = false;
    bool optimized = false;
}tmd;

// ============================================================================
// 2. Algebraic Operator Overloads
// ============================================================================

Dim operator+(const Dim& left, const Dim& right);
Dim operator*(const Dim& left, int64_t right_val);

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

// Internal helper for optimistic right-aligned shape dimension broadcasting
Dim broadcast_dims_optimistic(const Dim& a, const Dim& b);

// Protobuf extraction helpers
int64_t get_node_attr_int(const onnx::NodeProto& node, const std::string& name, int64_t default_val);
std::vector<int64_t> get_node_attr_ints(const onnx::NodeProto& node, const std::string& name);

// Helpers to print.
const char* data_type_to_string(DataType dtype);
const char* layout_to_string(Layout layout);

#endif // SHAPEINFER_H