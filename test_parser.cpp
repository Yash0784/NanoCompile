#include <iostream>
#include <fstream>
#include "onnx.pb.h" 

void getShape(onnx::ValueInfoProto value_info){
    std::cout << "Name: " << value_info.name() << " Shape: [";
    if(value_info.has_type() && value_info.type().has_tensor_type() && value_info.type().tensor_type().has_shape()){
        const auto& shape = value_info.type().tensor_type().shape();
        for(int i = 0; i < shape.dim_size(); i++){
            const auto& dim = shape.dim(i);

            if(dim.has_dim_value()){
                std::cout << dim.dim_value();
            }
            else if(dim.has_dim_param()){
                std::cout << dim.dim_param();
            }
            else{
                std::cout << '?';
            }

            if(i < shape.dim_size() - 1) std::cout << ",";
        }
        std::cout << "]\n";
    }
    else{
        std::cout << "Unavailable]\n";
    }
}for(onnx::ValueInfoProto out : outputs){
        getShape(out);
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
    
    for(int i = 0; i < graph.initializer_size(); ++i){
        // Access each individual tensor weight
        const onnx::TensorProto& tensor = graph.initializer(i);
        
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
        std::cout << node.name() << "\n";
        nodeCount++;
    }
    std::cout << "Total Nodes: " << nodeCount << "\n";

    // try getting shapes of all the tensors in the model;

    const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> inputs = graph.input();
    const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> outputs = graph.output();
    const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto> inters = graph.value_info();

    
    std::cout << "--------------------Graph inputs------------------------\n";

    for(onnx::ValueInfoProto in : inputs){
        getShape(in);
    }

    std::cout << "--------------------Graph outputs------------------------\n";

    for(onnx::ValueInfoProto out : outputs){
        getShape(out);
    }

    std::cout << "--------------------Graph inters------------------------\n";
    
    if(graph.value_info_size() == 0){
        std::cout << "No Intermediate Tensor data available\n";
    }

    for(onnx::ValueInfoProto inter : inters){
        getShape(inter);
    }

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}