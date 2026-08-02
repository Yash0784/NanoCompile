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


const char* data_type_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::UNKNOWN: return "UNKNOWN";
        case DataType::FLOAT32: return "FLOAT32";
        case DataType::UINT8:   return "UINT8";
        case DataType::INT8:    return "INT8";
        case DataType::INT32:   return "INT32";
        case DataType::INT64:   return "INT64";
        case DataType::FLOAT16: return "FLOAT16";
        default:                return "UNSUPPORTED_TYPE";
    }
}

const char* layout_to_string(Layout layout) {
    switch (layout) {
        case Layout::NCHW:         return "NCHW";
        case Layout::NHWC:         return "NHWC";
        case Layout::ROW_MAJOR:    return "ROW_MAJOR";
        case Layout::COLUMN_MAJOR: return "COLUMN_MAJOR";
        case Layout::FLAT:         return "FLAT";
        case Layout::UNSPECIFIED:  return "UNSPECIFIED";
        default:                   return "UNSUPPORTED_LAYOUT";
    }
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

// Helper to extract a string attribute (like auto_pad)
std::string get_node_attr_string(const onnx::NodeProto& node, const std::string& name) {
    for (const auto& attr : node.attribute()) {
        if (attr.name() == name && attr.has_s()) {
            return attr.s();
        }
    }
    return "";
}

// Helper function to extract integer values from an Initializer Tensor
std::vector<int64_t> get_initializer_int_values(const onnx::TensorProto& tensor) {
    std::vector<int64_t> values;
    
    // Check raw data first (most common for INT64 initializers)
    if (tensor.has_raw_data()) {
        const std::string& raw = tensor.raw_data();
        size_t count = raw.size() / sizeof(int64_t);
        const int64_t* data_ptr = reinterpret_cast<const int64_t*>(raw.data());
        for (size_t i = 0; i < count; ++i) {
            values.push_back(data_ptr[i]);
        }
    } 
    // Fallback if stored in the explicit int64 repeated field
    else if (tensor.int64_data_size() > 0) {
        for (int i = 0; i < tensor.int64_data_size(); ++i) {
            values.push_back(tensor.int64_data(i));
        }
    }
    // Fallback if stored in standard int32 field
    else if (tensor.int32_data_size() > 0) {
        for (int i = 0; i < tensor.int32_data_size(); ++i) {
            values.push_back(tensor.int32_data(i));
        }
    }
    return values;
}

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

void life_time_tens(const std::vector<onnx::NodeProto>& TopoNodes, std::unordered_map<std::string, tmd*>& master_tensor_map){
    
    //written assuming all the in names and out names are in the master tensor map;
    
    int i = 0;
    for(auto node : TopoNodes){

        for(auto in_name : node.input()){
            if(in_name.empty()) continue;

            tmd *meta = master_tensor_map[in_name];

            if(meta->first_use == -1){
                meta->first_use = meta->last_use = i;
                continue;
            }

            meta->last_use = i;

        }

        for(auto out_name : node.output()){
            if(out_name.empty()) continue;

            master_tensor_map[out_name]->born_at = i;
        }

        i++;
    }
}



void infer_relu(const onnx::NodeProto& node, const std::vector<tmd*>& inputs, std::vector<tmd*>& outputs){
    // Activation functions are completely element-wise.
    // The dimensions and data properties do not alter at all across the layer boundary.
    outputs[0]->shape = inputs[0]->shape;
    outputs[0]->dtype = inputs[0]->dtype;

    //Infer layout.
    if (inputs.empty() || outputs.empty()) return;
    outputs[0]->layout = inputs[0]->layout;
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

    //Infer layout.
    if (inputs.size() < 2 || outputs.empty()) return;
    
    // If one side is a structured image and the other is a generic unshaped bias vector,
    // the output keeps the image's structural layout layout state.
    if (inputs[0]->layout != Layout::UNSPECIFIED) {
        outputs[0]->layout = inputs[0]->layout;
    } else {
        outputs[0]->layout = inputs[1]->layout;
    }
}


void infer_matmul(const onnx::NodeProto& node, const std::vector<tmd*>& inputs, std::vector<tmd*>& outputs){
    const auto& shape_A = inputs[0]->shape;
    const auto& shape_B = inputs[1]->shape;
    
    size_t rA = shape_A.size();
    size_t rB = shape_B.size();

    if(rA == 0 || rB == 0){
        std::cerr << "[-] Error: rank of an input tensor is zero, infer_matmul returned.\n";
        return;
    }
    
    // Matrix Multiplication rule: Input A [..., M, K] x Input B [..., K, N] -> Output [..., M, N]
    Dim k1 = shape_A[rA - 1];
    Dim k2 = shape_B[rB - 2];

    if(std::holds_alternative<int64_t>(k1.expr) && std::holds_alternative<int64_t>(k2.expr)){
        if(k1.get_static() != k2.get_static()){
            std::cerr << "[-] Error: Incompatable tensor sizes, infer_gemm returned.\n";
            return;
        }
    }

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

    //Infer layout.
    if (inputs.empty() || outputs.empty()) return;
    
    if (inputs[0]->shape.size() == 2) {
        outputs[0]->layout = Layout::ROW_MAJOR;
    } else {
        outputs[0]->layout = Layout::UNSPECIFIED;
    }
}

void infer_gemm(const onnx::NodeProto& node, const std::vector<tmd*>& inputs, std::vector<tmd*>& outputs) {
    // 1. Safety check: Gemm must have at least inputs A and B
    if (inputs.size() < 2 || outputs.empty()) {
        std::cerr << "[-] Error: Gemm requires at least 2 inputs.\n";
        return;
    }

    const auto& shape_A = inputs[0]->shape;
    const auto& shape_B = inputs[1]->shape;

    // 2. Safety check: Ensure A and B actually have dimensions to read
    if (shape_A.size() < 2 || shape_B.size() < 2) {
        std::cerr << "[-] Error: Gemm inputs A and B must be 2D.\n";
        return;
    }

    // Read attributes transA and transB (default to 0 if not present)
    int64_t transA = get_node_attr_int(node, "transA", 0);
    int64_t transB = get_node_attr_int(node, "transB", 0);

    // Determine output dimensions safely based on transposition
    // This is likely where your operator[] crashed if shape_A or shape_B were smaller than expected!

    Dim k1 = (transA == 0) ? shape_A[1] : shape_A[0];
    Dim k2 = (transB == 0) ? shape_B[0] : shape_B[1];

    if(std::holds_alternative<int64_t>(k1.expr) && std::holds_alternative<int64_t>(k2.expr)){
        if(k1.get_static() != k2.get_static()){
            std::cerr << "[-] Error: Incompatable tensor sizes, infer_gemm returned.\n";
            return;
        }
    }

    Dim M = (transA == 0) ? shape_A[0] : shape_A[1];
    Dim N = (transB == 0) ? shape_B[1] : shape_B[0];

    // 3. Handle optional 3rd input C (Bias)
    if (inputs.size() > 2 && inputs[2] != nullptr){
        const auto& shape_C = inputs[2]->shape;
        bool flag = true;
        if(shape_C.size() == 1){
            if(shape_C[0].get_static() != N.get_static()) flag = false;
        }
        else if(shape_C.size() == 2){
            if(!((shape_C[0].get_static() == M.get_static() || shape_C[0].get_static() == 1) && (shape_C[1].get_static() == N.get_static() || (shape_C[1].get_static() == 1)))) flag = false;
        }
        else{
            flag = false;
        }

        if(!flag){
            std::cerr << "[-] Error: Bias shapee mismatch in gemm, infer_gemm returned.\n";
            return;
        }
    }

    // 4. Assign the inferred shape safely to the output
    outputs[0]->shape = {M, N};
    outputs[0]->dtype = inputs[0]->dtype; // Gemm preserves input data type

    //Infer layout.
    if (outputs.empty()) return;
    outputs[0]->layout = Layout::ROW_MAJOR;
}

void infer_conv(const onnx::NodeProto& node, const std::vector<tmd*>& inputs, std::vector<tmd*>& outputs) {
    if (inputs.size() < 2 || outputs.empty()) return;

    const auto& input_shape = inputs[0]->shape;   // [1, 1, 28, 28]
    const auto& weight_shape = inputs[1]->shape;  // [8, 1, 5, 5]

    if (input_shape.size() < 4 || weight_shape.size() < 4) return;

    size_t num_spatial_axes = input_shape.size() - 2;

    std::vector<int64_t> strides = get_node_attr_ints(node, "strides");
    if (strides.empty()) strides.assign(num_spatial_axes, 1);

    std::vector<int64_t> dilations = get_node_attr_ints(node, "dilations");
    if (dilations.empty()) dilations.assign(num_spatial_axes, 1);

    // Get the auto_pad string ("SAME_UPPER", "SAME_LOWER", "NOTSET", etc.)
    std::string auto_pad = get_node_attr_string(node, "auto_pad");

    std::vector<int64_t> pads = get_node_attr_ints(node, "pads");
    if (pads.empty()) pads.assign(num_spatial_axes * 2, 0);

    outputs[0]->shape.clear();
    outputs[0]->shape.push_back(input_shape[0]);  // Batch Size -> 1
    outputs[0]->shape.push_back(weight_shape[0]); // Out Channels -> 8

    for (size_t i = 0; i < num_spatial_axes; ++i) {
        if (input_shape[2 + i].is_static() && weight_shape[2 + i].is_static()) {
            int64_t in_size = input_shape[2 + i].get_static();
            int64_t stride  = strides[i];

            if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
                // FIXED: ONNX 'SAME' formula scales output directly by stride via ceiling division
                int64_t out_size = std::ceil(static_cast<double>(in_size) / stride);
                outputs[0]->shape.push_back(Dim(out_size));
            } else {
                // Fallback to explicit pads calculation if auto_pad is NOTSET or empty
                int64_t k_size  = weight_shape[2 + i].get_static();
                int64_t pad_begin = pads[i];
                int64_t pad_end   = pads[i + num_spatial_axes]; 
                int64_t dilation  = dilations[i];

                int64_t effective_k = (k_size - 1) * dilation + 1;
                int64_t out_size = std::floor(static_cast<double>(in_size + pad_begin + pad_end - effective_k) / stride) + 1;
                outputs[0]->shape.push_back(Dim(out_size));
            }
        } else {
            outputs[0]->shape.push_back(Dim("dynamic_spatial_out"));
        }
    }
    outputs[0]->dtype = inputs[0]->dtype;

    //Infer layout.
    if (inputs.empty() || outputs.empty()) return;
    
    // Check if an attribute explicitly requests a format change, otherwise pull input state
    std::string data_format = get_node_attr_string(node, "data_format"); 
    if (data_format == "NHWC") {
        outputs[0]->layout = Layout::NHWC;
    } else if (data_format == "NCHW") {
        outputs[0]->layout = Layout::NCHW;
    } else {
        // Safe default: propagate the exact layout configuration of the feature input
        outputs[0]->layout = inputs[0]->layout;
    }
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

    //Infer shape.
    if (inputs.empty() || outputs.empty()) return;
    outputs[0]->layout = inputs[0]->layout;
}

void infer_reshape(const onnx::NodeProto& node, const std::vector<tmd*>& inputs, std::vector<tmd*>& outputs, const google::protobuf::RepeatedPtrField<onnx::TensorProto>& initializers) {
    if (inputs.size() < 2 || outputs.empty()) return;

    tmd* data_tensor = inputs[0];
    tmd* shape_tensor = inputs[1];

    outputs[0]->dtype = data_tensor->dtype;
    outputs[0]->shape.clear();

    // 1. Look for the target shape initializer inside the graph's initializers bank
    bool found_static_shape = false;
    std::vector<int64_t> target_dims;

    for (const auto& init : initializers) {
        if (init.name() == shape_tensor->name) {
            target_dims = get_initializer_int_values(init);
            found_static_shape = !target_dims.empty();
            break;
        }
    }

    // 2. If we found the static dimensions, handle the ONNX Reshape rules (0 and -1)
    if (found_static_shape) {
        int64_t total_elements = 1;
        bool has_static_input = true;
        
        // Compute total elements in input shape to resolve a potential "-1" wildcard
        for (const auto& d : data_tensor->shape) {
            if (d.is_static()) {
                total_elements *= d.get_static();
            } else {
                has_static_input = false;
            }
        }

        int wildcard_index = -1;
        int64_t target_elements_product = 1;

        for (size_t i = 0; i < target_dims.size(); ++i) {
            int64_t dim = target_dims[i];

            if (dim == 0) {
                // ONNX Rule: '0' means copy the dimension from the input tensor at that index
                if (i < data_tensor->shape.size()) {
                    outputs[0]->shape.push_back(data_tensor->shape[i]);
                    if (data_tensor->shape[i].is_static()) {
                        target_elements_product *= data_tensor->shape[i].get_static();
                    }
                } else {
                    outputs[0]->shape.push_back(Dim("unknown"));
                }
            } 
            else if (dim == -1) {
                // ONNX Rule: '-1' means infer this dimension based on the remaining elements
                wildcard_index = static_cast<int>(i);
                outputs[0]->shape.push_back(Dim(-1)); // Placeholder
            } 
            else {
                outputs[0]->shape.push_back(Dim(dim));
                target_elements_product *= dim;
            }
        }

        // Resolve the wildcard (-1) dimension if both inputs and targets are static
        if (wildcard_index != -1 && has_static_input && target_elements_product > 0) {
            int64_t inferred_dim = total_elements / target_elements_product;
            outputs[0]->shape[wildcard_index] = Dim(inferred_dim);
        }

        return;
    }

    // Fallback: if it's genuinely dynamic, use the shape tensor's size as a fallback rank descriptor
    if (shape_tensor->shape.size() > 0 && shape_tensor->shape[0].is_static()) {
        int64_t rank = shape_tensor->shape[0].get_static();
        for (int64_t i = 0; i < rank; ++i) {
            outputs[0]->shape.push_back(Dim("dynamic_dim"));
        }
    } else {
        outputs[0]->shape.push_back(Dim("dynamic_reshape_rank"));
    }

    //Infer Layout.
    if (outputs.empty()) return;
    
    size_t out_rank = outputs[0]->shape.size();
    if (out_rank == 1) {
        outputs[0]->layout = Layout::FLAT;
    } else if (out_rank == 2) {
        outputs[0]->layout = Layout::ROW_MAJOR;
    } else {
        outputs[0]->layout = Layout::UNSPECIFIED; // Destroys semantic NCHW context
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

    //Infer layout.
    if (inputs.empty() || outputs.empty()) return;
    
    // If we have an NCHW layout and permute axes to [0, 2, 3, 1] -> it becomes NHWC
    if (inputs[0]->layout == Layout::NCHW && perm == std::vector<int64_t>{0, 2, 3, 1}) {
        outputs[0]->layout = Layout::NHWC;
    } else if (inputs[0]->layout == Layout::NHWC && perm == std::vector<int64_t>{0, 3, 1, 2}) {
        outputs[0]->layout = Layout::NCHW;
    } else {
        outputs[0]->layout = Layout::UNSPECIFIED; 
    }
    
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

    //Infer Layout.
    if (inputs.empty() || outputs.empty()) return;

    Layout baseline_layout = inputs[0]->layout;

    //Validate that all other tensors joining the party match the baseline
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i]->layout != baseline_layout) {
            std::cerr << "[-] Layout Mismatch Error: Concat inputs must have identical layouts!" << std::endl;
            outputs[0]->layout = Layout::UNSPECIFIED;
            return;
        }
    }

    outputs[0]->layout = baseline_layout;
}

void infershape_driver(const onnx::NodeProto& node, std::string op, const google::protobuf::RepeatedPtrField<onnx::TensorProto>& initializers, std::vector<tmd*>& node_inputs, std::vector<tmd*>& node_outputs){
    if (op == "Relu") {//tested
        infer_relu(node, node_inputs, node_outputs);
    }
    else if (op == "Add") {//tested
        infer_add(node, node_inputs, node_outputs);
    }
    else if (op == "MatMul") {//tested
        infer_matmul(node, node_inputs, node_outputs);
    }
    else if (op == "Gemm") {
        infer_gemm(node, node_inputs, node_outputs);
    }
    else if (op == "Conv") {//tested
        infer_conv(node, node_inputs, node_outputs);
    }
    else if (op == "MaxPool" || op == "AveragePool") {//tested
        infer_pooling(node, node_inputs, node_outputs);
                
        // Special secondary output handling for MaxPool tracking indices
        if (op == "MaxPool" && node_outputs.size() > 1) {
            node_outputs[1]->shape = node_outputs[0]->shape; // Indices share identical output dimensions
            node_outputs[1]->dtype = DataType::INT64;       // ONNX specification requires 64-bit integer tracking
        }
    }
    else if (op == "Reshape") {//tested
        infer_reshape(node, node_inputs, node_outputs, initializers);
    }
    else if (op == "Transpose") {
        infer_transpose(node, node_inputs, node_outputs);
    }
    else if (op == "Concat") {
        infer_concat(node, node_inputs, node_outputs);
    }
    else {
        // Safety handler to capture un-implemented layers instantly during graph parsing
        std::cerr << "[-] Error: Unsupported ONNX operator '" << op 
                << "' encountered on node '" << node.name() << "'." << "\n";
        // You can choose to throw an exception here depending on your runtime architecture requirements:
        // throw std::runtime_error("Unsupported operator: " + op);
    }
}
