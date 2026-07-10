#include <iostream>
#include <string>
#include <variant>
#include <fstream>
#include "onnx.pb.h" 
#include "shapeInfer.hpp"


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


// Helper function to print a symbolic expression recursively
void print_dim_expression(const Dim& dim) {
    if (std::holds_alternative<int64_t>(dim.expr)) {
        std::cout << std::get<int64_t>(dim.expr);
    } else if (std::holds_alternative<std::string>(dim.expr)) {
        std::cout << std::get<std::string>(dim.expr);
    } else {
        const auto& bin = std::get<Dim::BinaryOp>(dim.expr);
        std::cout << "(";
        print_dim_expression(*(bin.left));
        switch (bin.op) {
            case OpType::ADD: std::cout << " + "; break;
            case OpType::SUB: std::cout << " - "; break;
            case OpType::MUL: std::cout << " * "; break;
            case OpType::DIV: std::cout << " / "; break;
        }
        print_dim_expression(*(bin.right));
        std::cout << ")";
    }
}


DataType getDtype(const onnx::ValueInfoProto& value_info) {
    if (value_info.has_type() && value_info.type().has_tensor_type()) {
        
        int raw_type_int = value_info.type().tensor_type().elem_type();
        
        return static_cast<DataType>(raw_type_int);
    }
    
    return DataType::UNKNOWN;
}


std::vector<Dim> getShape(onnx::ValueInfoProto value_info){
    std::vector<Dim> ret;
    std::cout << "Name: " << value_info.name() << " Shape: [";
    if(value_info.has_type() && value_info.type().has_tensor_type() && value_info.type().tensor_type().has_shape()){
        const auto& shape = value_info.type().tensor_type().shape();
        for(int i = 0; i < shape.dim_size(); i++){
            const auto& dim = shape.dim(i);

            if(dim.has_dim_value()){
                ret.push_back(Dim{dim.dim_value()});
                std::cout << dim.dim_value();
            }
            else if(dim.has_dim_param()){
                ret.push_back(Dim{dim.dim_param()});
                std::cout << dim.dim_param();
            }
            else{
                ret.push_back(Dim{"?"});
                std::cout << '?';
            }

            if(i < shape.dim_size() - 1) std::cout << ",";
        }
        std::cout << "]\n";
    }
    else{
        std::cout << "Unavailable]\n";
    }
    return ret;
}

std::vector<Dim> get_initializer_shape(const onnx::TensorProto& initializer) {
    std::vector<Dim> shape;
    
    // initializer.dims_size() tells you the rank (number of dimensions)
    for (int i = 0; i < initializer.dims_size(); ++i) {
        // initializer.dims(i) reads the actual integer size of that axis
        int64_t static_dim = initializer.dims(i);
        shape.push_back(Dim(static_dim));
    }
    
    return shape;
}


void print_tensor_vector_metadata(const std::vector<tmd*>& tensors) {
    std::cout << "\n================= TENSOR METADATA SUMMARY =================\n";
    
    for (size_t i = 0; i < tensors.size(); ++i) {
        const tmd* tensor = tensors[i];
        if (!tensor) continue;

        // 1. Print Name (Appends " [INIT]" if the boolean flag is true)
        std::cout << "Tensor [" << i << "]: " << tensor->name;
        if (tensor->is_initializer) {
            std::cout << " [INIT]";
        }
        std::cout << "\n";

        // 2. Print Data Type
        std::cout << "  -> Type:  " << data_type_to_string(tensor->dtype) << "\n";
        
        // 3. Format and print the shape tracking vector
        std::cout << "  -> Shape: [";
        for (size_t j = 0; j < tensor->shape.size(); ++j) {
            print_dim_expression(tensor->shape[j]);
            if (j < tensor->shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
        std::cout << "-----------------------------------------------------------\n";
    }
}

int main(int argc, char* argv[]){
    if(argc < 2){
        std::cout << "Use this executable as ./exe_name <path_to_model>\n";
        return 1;
    }
    //verifying Protobuf runtime matches generated headers
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::ifstream input(argv[1], std::ios::in | std::ios::binary);

    if(!input){
        std::cerr << "Failed to open the ONNX file.\n";
        return 1;
    }

    //creating the model instance
    onnx::ModelProto model;
    if(!model.ParseFromIstream(&input)){
        std::cerr << "Failed to parse ONNX protobuf data.\n";
        return 1;
    }

    std::cout << "ONNX IR Version: " << model.ir_version() << "\n";
    std::cout << "Model Version: " << model.model_version() << "\n";
    std::cout << "Producer Name: " << model.producer_name() << "\n";
    std::cout << "Producer Version: "<< model.producer_version() << "\n";

    //reference to the Graph
    const onnx::GraphProto& graph = model.graph();

    //looping through the initializers using the _size() syntax
    std::cout << "\nTotal Initializers: " << graph.initializer_size() << "\n\n";

    std::unordered_map<std::string, bool> isInit;
    std::unordered_map<std::string, tmd*> master_tensor_map;
    std::vector<tmd*> tensors;
    
    for(int i = 0; i < graph.initializer_size(); ++i){
        // Access each individual tensor weight

        
        const onnx::TensorProto& tensor = graph.initializer(i);

        tmd* tens = new tmd;

        tens->name = tensor.name();
        tens->dtype = static_cast<DataType>(tensor.data_type());
        tens->shape = get_initializer_shape(tensor);
        tens->is_initializer = true;
        tens->is_constant = true;
        
        isInit[tensor.name()] = true;
        tensors.push_back(tens);
        master_tensor_map[tensor.name()] = tens;
        
        std::cout << "Initializer [" << i << "] Name: " << tensor.name() << "\n";
        std::cout << "  Dimensions: [";
        for (int j = 0; j < tensor.dims_size(); ++j) {
            std::cout << tensor.dims(j) << (j == tensor.dims_size() - 1 ? "" : ", ");
        }
        std::cout << "]\n\n";
    }

    const google::protobuf::RepeatedPtrField<onnx::NodeProto> nodes = graph.node();
    int nodeCount = 0;
    for(onnx::NodeProto node : nodes){
        std::cout << node.name() << " Inputs: ";
        for(auto& inName: node.input()){
            std::cout << inName << " ";
        }
        std::cout << " Outputs: ";
        for(auto& outName: node.output()){
            std::cout << outName << " ";
        }
        std::cout << "\n";
        nodeCount++;
    }
    std::cout << "Total Nodes: " << nodeCount << "\n";

    // try getting shapes of all the tensors in the model;

    const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> inputs = graph.input();
    const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> outputs = graph.output();
    const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> inters = graph.value_info();


    
    std::cout << "--------------------Graph inputs------------------------\n";

    for(onnx::ValueInfoProto in : inputs){
        tmd *tensor = new tmd();
        tensor->name = in.name();
        tensor->dtype = getDtype(in);
        std::cout << data_type_to_string(tensor->dtype) << " ";
        tensor->shape = getShape(in);
        tensors.push_back(tensor);
        master_tensor_map[in.name()] = tensor;
    }

    std::cout << "--------------------Graph inters------------------------\n";
    
    if(graph.value_info_size() == 0){
        std::cout << "No Intermediate Tensor data available Infering data\n";
        for (const auto& node : graph.node()){
        
            // Gather existing input pointers
            std::vector<tmd*> node_inputs;
            for (const auto& in_name : node.input()) {
                node_inputs.push_back(master_tensor_map[in_name]);
            }

            // Pre-allocate empty destination outputs based on node description
            std::vector<tmd*> node_outputs;
            for (const auto& out_name : node.output()) {
                tmd* new_output = new tmd();
                new_output->name = out_name;
                new_output->shape_inferred = true;
                
                // Map it immediately so subsequent nodes down the pipeline can reference it
                tensors.push_back(new_output);
                master_tensor_map[out_name] = new_output;
                node_outputs.push_back(new_output);
            }

            // STEP 4: Route the vectors to your shape inference functions
            std::string op = node.op_type();
            
            if (op == "Relu") {
                infer_relu(node, node_inputs, node_outputs);
            } 
            else if (op == "Add") {
                infer_add(node, node_inputs, node_outputs);
            }
            else if (op == "MatMul") {
                infer_matmul(node, node_inputs, node_outputs);
            }
            else if (op == "Gemm") {
                infer_gemm(node, node_inputs, node_outputs);
            }
            else if (op == "Conv") {
                infer_conv(node, node_inputs, node_outputs);
            }
            else if (op == "MaxPool" || op == "AveragePool") {
                infer_pooling(node, node_inputs, node_outputs);
                
                // Special secondary output handling for MaxPool tracking indices
                if (op == "MaxPool" && node_outputs.size() > 1) {
                    node_outputs[1]->shape = node_outputs[0]->shape; // Indices share identical output dimensions
                    node_outputs[1]->dtype = DataType::INT64;       // ONNX specification requires 64-bit integer tracking
                }
            }
            else if (op == "Reshape") {
                infer_reshape(node, node_inputs, node_outputs);
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
                        << "' encountered on node '" << node.name() << "'." << std::endl;
                // You can choose to throw an exception here depending on your runtime architecture requirements:
                // throw std::runtime_error("Unsupported operator: " + op);
            }
        
        }
    }

    for(onnx::ValueInfoProto inter : inters){
        if(isInit[inter.name()]) continue;
        tmd *tensor = new tmd();
        tensor->name = inter.name();
        tensor->dtype = getDtype(inter);
        std::cout << data_type_to_string(tensor->dtype) << " ";
        tensor->shape = getShape(inter);
        tensors.push_back(tensor);
        master_tensor_map[inter.name()] = tensor;
    }

    std::cout << "--------------------Graph outputs------------------------\n";

    for(onnx::ValueInfoProto out : outputs){
        tmd *tensor = new tmd();
        tensor->name = out.name();
        tensor->dtype = getDtype(out);
        std::cout << data_type_to_string(tensor->dtype) << " ";
        tensor->shape = getShape(out);
        tensors.push_back(tensor);
        master_tensor_map[out.name()] = tensor;
    }

    std::cout << "Total Tensors: " << tensors.size() << "\n";

    print_tensor_vector_metadata(tensors);


    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}