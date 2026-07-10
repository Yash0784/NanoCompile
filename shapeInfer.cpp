// infer function for every operator should be called at an operator and give the tmd struct which can be directly used
#include <vector>
#include <string>
#include <variant>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "onnx.pb.h"
#include "shapeInfer.hpp"

// Constructor for static dimensions (int64_t)
Dim::Dim(int64_t val) : expr(val) {}

// Constructor for dynamic string variables
Dim::Dim(std::string var) : expr(var) {}

// Constructor for binary operations (like additions or multiplications)
Dim::Dim(OpType op, Dim* l, Dim* r) : expr(BinaryOp{op, l, r}) {}

// Method to check if the dimension is a static constant number
bool Dim::is_static() const { 
    return std::holds_alternative<int64_t>(expr); 
}

// Method to safely retrieve the static integer value
int64_t Dim::get_static() const { 
    return std::get<int64_t>(expr); 
}

// --- Operator Overloads to Handle Equations with Objects Natively ---

// Generates a new Addition operation block or collapses constants immediately
Dim operator+(const Dim& left, const Dim& right){
    if (left.is_static() && right.is_static()) {
        return Dim(left.get_static() + right.get_static());
    }
    // Allocate heap structures to store variable equations safely
    return Dim(OpType::ADD, new Dim(left), new Dim(right));
}

// Generates a new Multiplication expression tracking block (e.g., height * 2)
Dim operator*(const Dim& left, int64_t right_val) {
    if (left.is_static()) return Dim(left.get_static() * right_val);
    return Dim(OpType::MUL, new Dim(left), new Dim(right_val));
}

// --- Protobuf Attribute Parsing Extraction Utilities ---
int64_t get_node_attr_int(const onnx::NodeProto& node, const std::string& name, int64_t default_val){
    for (const auto& attr : node.attribute()) {
        if (attr.name() == name) return attr.i();
    }
    return default_val;
}

std::vector<int64_t> get_node_attr_ints(const onnx::NodeProto& node, const std::string& name){
    std::vector<int64_t> vals;
    for (const auto& attr : node.attribute()) {
        if (attr.name() == name) {
            for (int i = 0; i < attr.ints_size(); ++i) vals.push_back(attr.ints(i));
            break;
        }
    }
    return vals;
}

void infer_relu(const onnx::NodeProto& node, const std::vector<tmd*>& inputs, std::vector<tmd*>& outputs){
    // Activation functions are completely element-wise.
    // The dimensions and data properties do not alter at all across the layer boundary.
    outputs[0]->shape = inputs[0]->shape;
    outputs[0]->dtype = inputs[0]->dtype;
}


// Evaluates which dimension wins the broadcasting matching test under an optimistic assumption
Dim broadcast_dims_optimistic(const Dim& a, const Dim& b){
    // If input A is explicitly a unit channel 1, B is the expanded shape dimension
    if (a.is_static() && a.get_static() == 1) return b;
    // If input B is explicitly a unit channel 1, A is the expanded shape dimension
    if (b.is_static() && b.get_static() == 1) return a;
    
    // Optimistic fallback: Since the exported graph is valid, if neither is 1,
    // they must be identical values at execution runtime. Return either option.
    return a; 
}

void infer_add(const onnx::NodeProto& node, const std::vector<tmd*>& inputs, std::vector<tmd*>& outputs){
    const auto& shape_A = inputs[0]->shape;
    const auto& shape_B = inputs[1]->shape;
    
    // Broadcasting iterates from back-to-front, so the maximum rank determines the output vector size
    size_t max_rank = std::max(shape_A.size(), shape_B.size());
    std::vector<Dim> out_shape;
    out_shape.reserve(max_rank);

    // Step from the back of the shape arrays toward the front (right-aligned broadcast logic)
    for (size_t i = 0; i < max_rank; ++i) {
        // Pad with a unit dimension of 1 if one tensor has a smaller rank than the other
        Dim dim_A = (i < shape_A.size()) ? shape_A[shape_A.size() - 1 - i] : Dim(1);
        Dim dim_B = (i < shape_B.size()) ? shape_B[shape_B.size() - 1 - i] : Dim(1);
        
        // Push to the front of our processing list to maintain correct shape directionality
        out_shape.insert(out_shape.begin(), broadcast_dims_optimistic(dim_A, dim_B));
    }
    outputs[0]->shape = out_shape;
    outputs[0]->dtype = inputs[0]->dtype;
}


void infer_matmul(const onnx::NodeProto& node, const std::vector<tmd*>& inputs, std::vector<tmd*>& outputs){
    const auto& shape_A = inputs[0]->shape;
    const auto& shape_B = inputs[1]->shape;
    
    size_t rA = shape_A.size();
    size_t rB = shape_B.size();
    
    // Matrix Multiplication rule: Input A [..., M, K] x Input B [..., K, N] -> Output [..., M, N]
    Dim M = (rA > 1) ? shape_A[rA - 2] : Dim(1);
    Dim N = shape_B[rB - 1];

    // Compute dynamic layout alignments for outer batch channels (e.g., 3D/4D tensors)
    size_t batch_rank = std::max(rA > 2 ? rA - 2 : 0, rB > 2 ? rB - 2 : 0);
    std::vector<Dim> out_shape;
    
    for (size_t i = 0; i < batch_rank; ++i) {
        Dim dim_A = (i < rA - 2) ? shape_A[rA - 3 - i] : Dim(1);
        Dim dim_B = (i < rB - 2) ? shape_B[rB - 3 - i] : Dim(1);
        out_shape.insert(out_shape.begin(), broadcast_dims_optimistic(dim_A, dim_B));
    }
    
    // Append structural matrix parameters to finalize the vector
    if (rA > 1) out_shape.push_back(M);
    out_shape.push_back(N);

    outputs[0]->shape = out_shape;
    outputs[0]->dtype = inputs[0]->dtype;
}

void infer_gemm(const onnx::NodeProto& node, const std::vector<tmd*>& inputs, std::vector<tmd*>& outputs) {
    // 1. Safety check: Gemm must have at least inputs A and B
    if (inputs.size() < 2 || outputs.empty()) {
        std::cerr << "[-] Error: Gemm requires at least 2 inputs." << std::endl;
        return;
    }

    const auto& shape_A = inputs[0]->shape;
    const auto& shape_B = inputs[1]->shape;

    // 2. Safety check: Ensure A and B actually have dimensions to read
    if (shape_A.size() < 2 || shape_B.size() < 2) {
        std::cerr << "[-] Error: Gemm inputs A and B must be at least 2D." << std::endl;
        return;
    }

    // Read attributes transA and transB (default to 0 if not present)
    int64_t transA = get_node_attr_int(node, "transA", 0);
    int64_t transB = get_node_attr_int(node, "transB", 0);

    // Determine output dimensions safely based on transposition
    // This is likely where your operator[] crashed if shape_A or shape_B were smaller than expected!
    Dim M = (transA == 0) ? shape_A[0] : shape_A[1];
    Dim N = (transB == 0) ? shape_B[1] : shape_B[0];

    // 3. Handle optional 3rd input C (Bias)
    if (inputs.size() > 2 && inputs[2] != nullptr) {
        const auto& shape_C = inputs[2]->shape;
        // You can add validation logic for shape_C here if needed, 
        // but crucially, do NOT assume shape_C[0] is safe without checking shape_C.size()!
    }

    // 4. Assign the inferred shape safely to the output
    outputs[0]->shape = {M, N};
    outputs[0]->dtype = inputs[0]->dtype; // Gemm preserves input data type
}

void infer_conv(const onnx::NodeProto& node, const std::vector<tmd*>& inputs, std::vector<tmd*>& outputs){
    // Input 0 layout: [Batch_Size, Input_Channels, Height, Width]
    // Input 1 layout (Weights): [Output_Channels, Input_Channels, Kernel_H, Kernel_W]
    Dim N = inputs[0]->shape[0];
    Dim M = inputs[1]->shape[0]; 

    // Extract operational properties governing size scaling
    std::vector<int64_t> strides = get_node_attr_ints(node, "strides");
    std::vector<int64_t> pads = get_node_attr_ints(node, "pads");
    std::vector<int64_t> dilations = get_node_attr_ints(node, "dilations");

    std::vector<Dim> out_shape = { N, M };
    
    // Dynamically iterate through variable spatial axes (supporting both 1D, 2D, or 3D Convolutions)
    for (size_t i = 2; i < inputs[0]->shape.size(); ++i) {
        auto in_dim = inputs[0]->shape[i];
        auto k_dim = inputs[1]->shape[i];

        // Perform numerical sampling analysis if the current layout components are static integers
        if (in_dim.is_static() && k_dim.is_static()) {
            int64_t stride = (strides.size() > i - 2) ? strides[i - 2] : 1;
            int64_t pad_low = (pads.size() > (i - 2)) ? pads[i - 2] : 0;
            int64_t pad_high = (pads.size() > (i - 2) + (inputs[0]->shape.size() - 2)) ? pads[(i - 2) + (inputs[0]->shape.size() - 2)] : 0;
            int64_t dilation = (dilations.size() > i - 2) ? dilations[i - 2] : 1;

            // Mathematical standard Convolution Output Dimension formula
            int64_t out_dim = ((in_dim.get_static() + pad_low + pad_high - dilation * (k_dim.get_static() - 1) - 1) / stride) + 1;
            out_shape.push_back(Dim(out_dim));
        } else {
            // Generate a labeled dynamic fallback string tracker if dimensions are variable strings
            out_shape.push_back(Dim(inputs[0]->name + "_conv_spatial"));
        }
    }
    outputs[0]->shape = out_shape;
    outputs[0]->dtype = inputs[0]->dtype;
}

void infer_pooling(const onnx::NodeProto& node, const std::vector<tmd*>& inputs, std::vector<tmd*>& outputs){
    // Pooling drops spatial width/height features but keeps total channel volume identical
    Dim N = inputs[0]->shape[0];
    Dim C = inputs[0]->shape[1]; 

    std::vector<int64_t> kernel_shape = get_node_attr_ints(node, "kernel_shape");
    std::vector<int64_t> strides = get_node_attr_ints(node, "strides");
    std::vector<int64_t> pads = get_node_attr_ints(node, "pads");

    std::vector<Dim> out_shape = { N, C };

    // Calculate structural compression limits sequentially across spatial axes
    for (size_t i = 2; i < inputs[0]->shape.size(); ++i) {
        auto in_dim = inputs[0]->shape[i];
        if (in_dim.is_static() && kernel_shape.size() > i - 2) {
            int64_t k_dim = kernel_shape[i - 2];
            int64_t stride = (strides.size() > i - 2) ? strides[i - 2] : 1;
            int64_t pad_low = (pads.size() > (i - 2)) ? pads[i - 2] : 0;
            int64_t pad_high = (pads.size() > (i - 2) + 2) ? pads[(i - 2) + 2] : 0;

            // Pooling explicit scaling math
            int64_t out_dim = ((in_dim.get_static() + pad_low + pad_high - k_dim) / stride) + 1;
            out_shape.push_back(Dim(out_dim));
        } else {
            out_shape.push_back(Dim(inputs[0]->name + "_pool_spatial"));
        }
    }
    outputs[0]->shape = out_shape;
    outputs[0]->dtype = inputs[0]->dtype;
}

void infer_reshape(const onnx::NodeProto& node, const std::vector<tmd*>& inputs, std::vector<tmd*>& outputs){
    outputs[0]->dtype = inputs[0]->dtype;
    
    // Reshape uses input[1] as a 1D tensor tracking the target destination shape description
    if (inputs.size() > 1 && inputs[1]->data != nullptr) {
        int64_t* target_shape_data = static_cast<int64_t*>(inputs[1]->data);
        size_t target_rank = inputs[1]->shape[0].get_static(); 

        std::vector<Dim> out_shape;
        int64_t infer_axis = -1;

        for (size_t i = 0; i < target_rank; ++i) {
            int64_t requested_dim = target_shape_data[i];
            
            if (requested_dim == 0) {
                // ONNX Rule: '0' means copy the dimension size directly from the source tensor
                out_shape.push_back(inputs[0]->shape[i]); 
            } else if (requested_dim == -1) {
                // ONNX Rule: '-1' means automatically compute this axis based on total remaining element math
                infer_axis = i;
                out_shape.push_back(Dim("-1_placeholder")); 
            } else {
                // Use the explicit layout target configuration constant specified
                out_shape.push_back(Dim(requested_dim));
            }
        }

        // If a -1 placeholder exists, compute it by checking total source vs destination elements
        if (infer_axis != -1) {
            bool total_static = true;
            int64_t source_elements = 1;
            for (const auto& d : inputs[0]->shape) {
                if (d.is_static()) source_elements *= d.get_static(); else total_static = false;
            }
            
            int64_t target_elements = 1;
            for (size_t i = 0; i < out_shape.size(); ++i) {
                if (i != static_cast<size_t>(infer_axis) && out_shape[i].is_static()) {
                    target_elements *= out_shape[i].get_static();
                }
            }

            // Successfully overwrite the placeholder if everything parses out statically
            if (total_static && target_elements > 0) {
                out_shape[infer_axis] = Dim(source_elements / target_elements);
            }
        }
        outputs[0]->shape = out_shape;
    } else {
        // Fallback for purely dynamic graphs where targets change per evaluation slice
        outputs[0]->shape = { Dim("dynamic_reshape_rank") };
    }
}

void infer_transpose(const onnx::NodeProto& node, const std::vector<tmd*>& inputs, std::vector<tmd*>& outputs) {
    std::vector<int64_t> perm = get_node_attr_ints(node, "perm");
    size_t rank = inputs[0]->shape.size();
    std::vector<Dim> out_shape(rank, Dim(0)); // Allocate slot positions

    if (perm.empty()) {
        // ONNX Specification: If 'perm' attribute is empty, completely reverse the axis layout stack
        for (size_t i = 0; i < rank; ++i) {
            out_shape[i] = inputs[0]->shape[rank - 1 - i]; 
        }
    } else {
        // Remap dimensions following the custom permutation indices specified (e.g., {0, 2, 3, 1})
        for (size_t i = 0; i < rank; ++i) {
            out_shape[i] = inputs[0]->shape[perm[i]];
        }
    }
    outputs[0]->shape = out_shape;
    outputs[0]->dtype = inputs[0]->dtype;
}

void infer_concat(const onnx::NodeProto& node, const std::vector<tmd*>& inputs, std::vector<tmd*>& outputs){
    int64_t axis = get_node_attr_int(node, "axis", 0);
    size_t rank = inputs[0]->shape.size();
    
    // Support wrapping for negative indexing parameters (e.g., axis -1 points to last dimension)
    if (axis < 0) axis += rank;

    // Use the first input to baseline standard architectural dimensions
    std::vector<Dim> out_shape = inputs[0]->shape; 
    Dim concat_axis_expr = inputs[0]->shape[axis];

    // Iteratively append sizes to accumulate the target connection axis symbolically
    for (size_t i = 1; i < inputs.size(); ++i) {
        concat_axis_expr = concat_axis_expr + inputs[i]->shape[axis]; // Invokes our overloaded '+' operator
    }

    out_shape[axis] = concat_axis_expr;
    outputs[0]->shape = out_shape;
    outputs[0]->dtype = inputs[0]->dtype;
}


