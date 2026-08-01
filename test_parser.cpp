#include <iostream>
#include <string>
#include <variant>
#include <fstream>
#include "onnx.pb.h" 
#include "shapeInfer.hpp"
//#include "graph.hpp"



class ONNXGraph {
public:
    // Model Metadata
    int64_t ir_version = 0;
    int64_t model_version = 0;
    std::string producer_name = "Unknown";
    std::string producer_version = "Unknown";
    std::string domain = "";

    // Nodes and edges
    std::unordered_map<std::string, bool> isInit;
    std::unordered_map<std::string, tmd*> master_tensor_map;
    std::vector<tmd*> tensors;
    std::vector<onnx::NodeProto> nodes;
    void build_graph(const onnx::ModelProto& model);
    //void print_metadata(std::ostream& os) const;
};




// Helper function to print a symbolic expression recursively
void print_dim_expression(const Dim& dim, std::ostream& os = std::cout) {
    if (std::holds_alternative<int64_t>(dim.expr)) {
        os << std::get<int64_t>(dim.expr);
    } else if (std::holds_alternative<std::string>(dim.expr)) {
        os << std::get<std::string>(dim.expr);
    } else {
        const auto& bin = std::get<Dim::BinaryOp>(dim.expr);
        os << "(";
        // Pass 'os' recursively so nested expressions write to the same stream
        print_dim_expression(*(bin.left), os);
        switch (bin.op) {
            case OpType::ADD: os << " + "; break;
            case OpType::SUB: os << " - "; break;
            case OpType::MUL: os << " * "; break;
            case OpType::DIV: os << " / "; break;
        }
        print_dim_expression(*(bin.right), os);
        os << ")";
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

//Guess input Layout, might need changes for a better compiler.

Layout getLayout(tmd* tensor){
    if(tensor->shape.size() == 4) return Layout::NCHW;
    else if(tensor->shape.size() == 2) return Layout::ROW_MAJOR;
    else if(tensor->shape.size() == 1) return Layout::FLAT;
    else return Layout::UNSPECIFIED;
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


void print_tensor_vector_metadata(const std::vector<tmd*>& tensors, std::ostream& os = std::cout) {
    os << "\n================= TENSOR METADATA SUMMARY =================\n";
    
    for (size_t i = 0; i < tensors.size(); ++i) {
        const tmd* tensor = tensors[i];
        if (!tensor) continue;

        // 1. Print Name (Appends " [INIT]" if the boolean flag is true)
        os << "Tensor [" << i << "]: " << tensor->name;
        if (tensor->is_initializer) {
            os << " [INIT]";
        }
        os << "\n";

        // 2. Print Data Type
        os << "  -> Type:      " << data_type_to_string(tensor->dtype) << "\n";
        
        // 3. Format and print the shape tracking vector
        os << "  -> Shape:     [";
        for (size_t j = 0; j < tensor->shape.size(); ++j) {
            print_dim_expression(tensor->shape[j], os);
            if (j < tensor->shape.size() - 1) {
                os << ", ";
            }
        }
        os << "]\n";
        os << "  -> Layout:    " << layout_to_string(tensor->layout) << "\n";

        // 4. Print Producer Node
        os << "  -> Producer:  ";
        if (tensor->producer != nullptr) {
            // Display node name if set, otherwise fallback to its op_type
            std::string prod_name = tensor->producer->name().empty() 
                ? ("Node(" + tensor->producer->op_type() + ")") 
                : tensor->producer->name();
            os << prod_name << " [" << tensor->producer->op_type() << "]\n";
        } else {
            // Tensors with no producer are graph inputs or static weight initializers
            os << (tensor->is_initializer ? "None [WEIGHT_INITIALIZER]" : "None [GRAPH_INPUT]") << "\n";
        }

        // 5. Print Consumer Nodes
        os << "  -> Consumers: ";
        if (tensor->consumers.empty()) {
            os << "None [GRAPH_OUTPUT]\n";
        } else {
            os << "[";
            for (size_t k = 0; k < tensor->consumers.size(); ++k) {
                const auto* consumer = tensor->consumers[k];
                if (!consumer) continue;

                std::string cons_name = consumer->name().empty() 
                    ? ("Node(" + consumer->op_type() + ")") 
                    : consumer->name();

                os << cons_name << " [" << consumer->op_type() << "]";
                if (k < tensor->consumers.size() - 1) {
                    os << ", ";
                }
            }
            os << "]\n";
        }

        os << "-----------------------------------------------------------\n";
    }
}

void tensor_uninfered(tmd *tensor, onnx::ValueInfoProto tens, bool isInput, std::unordered_map<std::string, tmd*>& master_tensor_map, std::vector<tmd*>& tensors){
    tensor->name = tens.name();
    tensor->dtype = getDtype(tens);
    std::cout << data_type_to_string(tensor->dtype) << " ";
    tensor->shape = getShape(tens);
    if(isInput) tensor->layout = getLayout(tensor);
    tensors.push_back(tensor);
    master_tensor_map[tens.name()] = tensor;
}

void link_tensor_producers_and_consumers(const std::vector<onnx::NodeProto>& topo_nodes,std::unordered_map<std::string, tmd*>& master_tensor_map) 
{
    // Traverse through topologically sorted nodes
    for (const auto& node : topo_nodes) {

        // 1. PROCESS INPUTS -> Record this node as a CONSUMER of input tensors
        for (const auto& input_name : node.input()) {
            // ONNX allows empty strings for optional inputs (e.g., missing Conv bias)
            if (input_name.empty()) continue;

            auto it = master_tensor_map.find(input_name);
            if (it != master_tensor_map.end() && it->second != nullptr) {
                tmd* input_tensor = it->second;
                
                // Add current node pointer to this tensor's consumer list
                input_tensor->consumers.push_back(&node);
            }
        }

        // 2. PROCESS OUTPUTS -> Record this node as the PRODUCER of output tensors
        for (const auto& output_name : node.output()) {
            if (output_name.empty()) continue;

            auto it = master_tensor_map.find(output_name);
            if (it != master_tensor_map.end() && it->second != nullptr) {
                tmd* output_tensor = it->second;
                
                // Link current node pointer as the producer of this tensor
                output_tensor->producer = &node;
            }
        }
    }
}

void ONNXGraph::build_graph(const onnx::ModelProto& model){
    // Extract Global Metadata
    this->ir_version = model.ir_version();
    this->model_version = model.model_version();
    this->producer_name = model.producer_name();
    this->producer_version = model.producer_version();
    this->domain = model.domain();
    const onnx::GraphProto& graph = model.graph();

    for(int i = 0; i < graph.initializer_size(); ++i){
        // Access each individual tensor weight
        const onnx::TensorProto& tensor = graph.initializer(i);

        tmd* tens = new tmd;

        tens->name = tensor.name();
        tens->dtype = static_cast<DataType>(tensor.data_type());
        tens->shape = get_initializer_shape(tensor);
        tens->layout = getLayout(tens);
        tens->is_initializer = true;
        tens->is_constant = true;

        isInit[tensor.name()] = true;
        tensors.push_back(tens);
        master_tensor_map[tensor.name()] = tens;
    }

    nodes.clear();
    nodes.reserve(graph.node_size());

    for (const auto& node : graph.node()) {
        nodes.push_back(node);
    }

    const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> inputs = graph.input();
    const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> outputs = graph.output();
    const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> inters = graph.value_info();

    for(onnx::ValueInfoProto in : inputs){
        tmd *tensor = new tmd();
        tensor_uninfered(tensor, in, true, master_tensor_map, tensors);
    }

    for(onnx::ValueInfoProto out : outputs){
        tmd *tensor = new tmd();
        tensor_uninfered(tensor, out, false, master_tensor_map, tensors);
    }

    for(onnx::ValueInfoProto inter : inters){
        if(isInit[inter.name()]) continue;
        tmd *tensor = new tmd();
        tensor_uninfered(tensor, inter, false, master_tensor_map, tensors);
    }

    if(graph.value_info_size() == 0){
        std::cout << "No Intermediate Tensor data available Infering data\n";
        for (const auto& node : nodes){
        
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

            // Route the vectors to shape inference functions
            infershape_driver(node, node.op_type(), graph.initializer(), node_inputs, node_outputs);
        
        }
    }
    link_tensor_producers_and_consumers(nodes, master_tensor_map);
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

    //reference to the Graph
    ONNXGraph graph;
    
    graph.build_graph(model);

    std::ofstream outfile("tensor_summary.txt");
    if (outfile.is_open()) {
        print_tensor_vector_metadata(graph.tensors, outfile);
        outfile.close(); // <-- ADD THIS: Ensures memory buffer flushes completely to disk!
        std::cout << "Successfully saved tensor summary to tensor_summary.txt\n";
    }
    else {
        std::cerr << "Error: Could not open output file for writing.\n";
    }

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}