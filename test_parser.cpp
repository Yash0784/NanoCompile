#include "graph.hpp"

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