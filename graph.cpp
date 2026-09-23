#include <iostream>
#include <string>
#include <fstream>
#include "onnx.pb.h" 
#include "graph.hpp"
#include "shapeInfer.hpp"


DataType getDtype(const onnx::ValueInfoProto& value_info) {
    if (value_info.has_type() && value_info.type().has_tensor_type()) {
        
        int raw_type_int = value_info.type().tensor_type().elem_type();
        
        return static_cast<DataType>(raw_type_int);
    }
    
    return DataType::UNKNOWN;
}


// Reads a ValueInfoProto's shape and returns it as a fully concrete
// vector<int64_t>. ONNX represents each axis as EITHER a fixed integer
// (dim_value) OR a symbolic name (dim_param, e.g. "batch_size", "N" --
// exporters aren't consistent about the string). Since this project fixes
// its batch size before parsing (see ONNXGraph::build_graph), every
// dim_param encountered here is substituted with `batch_size` directly.
//
// NOTE: this assumes the only symbolic axis your model exports is the
// batch axis. That's true for a plain feed-forward/CNN inference graph
// (the common case), but if a model genuinely has some OTHER dynamic axis
// (e.g. a variable sequence length), substituting batch_size for it would
// silently produce the wrong shape. The printed line below every time a
// substitution happens exists so you can eyeball the dim_param names in
// your own model and confirm this assumption actually holds before trusting
// the output.
std::vector<int64_t> getShape(onnx::ValueInfoProto value_info, int64_t batch_size){
    std::vector<int64_t> ret;
    std::cout << "Name: " << value_info.name() << " Shape: [";
    if(value_info.has_type() && value_info.type().has_tensor_type() && value_info.type().tensor_type().has_shape()){
        const auto& shape = value_info.type().tensor_type().shape();
        for(int i = 0; i < shape.dim_size(); i++){
            const auto& dim = shape.dim(i);

            if(dim.has_dim_value()){
                ret.push_back(dim.dim_value());
                std::cout << dim.dim_value();
            }
            else if(dim.has_dim_param()){
                // Symbolic axis -- substitute the fixed batch size.
                ret.push_back(batch_size);
                std::cout << batch_size << " (resolved from symbolic '" << dim.dim_param() << "')";
            }
            else{
                // Neither a fixed value nor a named symbol -- ONNX allows
                // a totally empty dim entry for "unknown", which we can't
                // resolve to anything meaningful. Flag it rather than
                // guessing.
                std::cerr << "\n[-] Warning: tensor '" << value_info.name()
                          << "' has an unresolvable dimension at axis " << i
                          << " (no dim_value, no dim_param). Defaulting to 1.\n";
                ret.push_back(1);
                std::cout << '1';
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

// Initializer dims in ONNX are always concrete integers -- a weight tensor
// can never have a symbolic axis -- so this needs no batch_size argument
// and no substitution logic.
std::vector<int64_t> get_initializer_shape(const onnx::TensorProto& initializer) {
    std::vector<int64_t> shape;
    
    // initializer.dims_size() tells you the rank (number of dimensions)
    for (int i = 0; i < initializer.dims_size(); ++i) {
        // initializer.dims(i) reads the actual integer size of that axis
        int64_t static_dim = initializer.dims(i);
        shape.push_back(static_dim);
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
        
        // 3. Format and print the shape vector (plain integers now, no
        //    symbolic expression tree to walk).
        os << "  -> Shape:     [";
        for (size_t j = 0; j < tensor->shape.size(); ++j) {
            os << tensor->shape[j];
            if (j < tensor->shape.size() - 1) {
                os << ", ";
            }
        }
        os << "]\n";
        os << "  -> Layout:    " << layout_to_string(tensor->layout) << "\n";

        // 4. Byte size -- this is the new information the swap planner
        //    actually needs: how much GPU memory this one tensor costs.
        os << "  -> Elements:  " << tensor->num_elements << "\n";
        os << "  -> Bytes:     " << tensor->bytes
           << " (" << (tensor->bytes / 1024.0 / 1024.0) << " MB)\n";

        // 5. Print Producer Node
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

        // 6. Print Consumer Nodes
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

        os << "  -> Born_at: " << tensor->born_at << "  First_use: " << tensor->first_use << "  Last_use: " << tensor->last_use << "\n";

        os << "-----------------------------------------------------------\n";
    }
}

void tensor_uninfered(tmd *tensor, onnx::ValueInfoProto tens, bool isInput, std::unordered_map<std::string, tmd*>& master_tensor_map, std::vector<tmd*>& tensors, int64_t batch_size){
    tensor->name = tens.name();
    tensor->dtype = getDtype(tens);
    std::cout << data_type_to_string(tensor->dtype) << " ";
    tensor->shape = getShape(tens, batch_size);
    if(isInput) tensor->layout = getLayout(tensor);

    // Shape is fully concrete the moment getShape returns, so bytes can be
    // computed right here instead of in a separate pass over the graph.
    compute_tensor_bytes(tensor);

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

void ONNXGraph::build_graph(const onnx::ModelProto& model, int64_t batch_size){
    // Lock in the batch size this whole graph (and every byte-size number
    // derived from it) is valid for. See the header comment on
    // ONNXGraph::batch_size for what re-planning for a different batch
    // size requires.
    this->batch_size = batch_size;

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

        // Initializer shapes are always concrete already (no symbolic axis
        // is legal on a weight tensor), so bytes can be computed immediately.
        compute_tensor_bytes(tens);

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
        tensor_uninfered(tensor, in, true, master_tensor_map, tensors, batch_size);
    }

    for(onnx::ValueInfoProto out : outputs){
        tmd *tensor = new tmd();
        tensor_uninfered(tensor, out, false, master_tensor_map, tensors, batch_size);
    }

    for(onnx::ValueInfoProto inter : inters){
        if(isInit[inter.name()]) continue;
        tmd *tensor = new tmd();
        tensor_uninfered(tensor, inter, false, master_tensor_map, tensors, batch_size);
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

            // Route the vectors to shape inference functions. Byte-size
            // computation for each of node_outputs happens inside
            // infershape_driver itself, right after the shape is set.
            infershape_driver(node, node.op_type(), graph.initializer(), node_inputs, node_outputs);
        
        }
    }
    link_tensor_producers_and_consumers(nodes, master_tensor_map);

    life_time_tens(nodes, master_tensor_map);

}
