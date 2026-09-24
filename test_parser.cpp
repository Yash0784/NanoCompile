#include "graph.hpp"
#include "op_profiler.hpp" // profile_node() + the cuda/cublas/cudnn types it needs
#include "swap_plan.hpp"

int main(int argc, char* argv[]){
    if(argc < 2){
        std::cout << "Use this executable as ./exe_name <path_to_model> [batch_size]\n";
        return 1;
    }

    int64_t batch_size = 1;
    if(argc >= 3){
        batch_size = std::stoll(argv[2]);
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
    graph.build_graph(model, batch_size); // every tensor's shape/bytes are ready after this returns


    cudaStream_t compute, swap_in, swap_out;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&swap_in, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&swap_out, cudaStreamNonBlocking));

    cublasHandle_t g_cublas; CUBLAS_CHECK(cublasCreate(&g_cublas));
    cudnnHandle_t  g_cudnn;  CUDNN_CHECK(cudnnCreate(&g_cudnn));
    CUBLAS_CHECK(cublasSetStream(g_cublas, compute));
    CUDNN_CHECK(cudnnSetStream(g_cudnn, compute));

    // One-time PCIe bandwidth calibration, on the exact swap_in/swap_out
    // streams the real execution will later use, then derive every
    // tensor's swap_in_time/swap_out_time from its real ->bytes.
    PCIeBandwidthModel pcie_model = profile_pcie_bandwidth(swap_in, swap_out);
    compute_tensor_swap_times(graph.tensors, pcie_model);

    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        const auto& node = graph.nodes[i];

        std::vector<tmd*> node_inputs;
        for (const auto& in_name : node.input()) {
            if (in_name.empty()) continue; // optional inputs (e.g. missing Conv bias)
            node_inputs.push_back(graph.master_tensor_map[in_name]);
        }

        std::vector<tmd*> node_outputs;
        for (const auto& out_name : node.output()) {
            node_outputs.push_back(graph.master_tensor_map[out_name]);
        }

        double ms = profile_node(node, node.op_type(), node_inputs, node_outputs,
                                  g_cublas, g_cudnn, compute);

        graph.nodeTime.push_back(ms);

        std::cout << "[" << i << "] " << node.op_type()
                  << " (" << node.name() << "): " << ms << " ms\n";
    }

    CUDNN_CHECK(cudnnDestroy(g_cudnn));
    CUBLAS_CHECK(cublasDestroy(g_cublas));
    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaStreamDestroy(swap_in));
    CUDA_CHECK(cudaStreamDestroy(swap_out));

    // ---------------------------------------------------------------
    // Existing tensor metadata dump -- unchanged. print_tensor_vector_
    // metadata (in graph.cpp) doesn't print exec_ms yet since I don't
    // have your current copy of that file with the field added -- add
    // one line there (`os << "  -> exec_ms: " << tensor->exec_ms << "\n";`
    // next to the existing Bytes/Elements lines) if you want it in
    // tensor_summary.txt too.
    // ---------------------------------------------------------------

    std::ofstream outfile("tensor_summary.txt");
    if (outfile.is_open()) {
        print_tensor_vector_metadata(graph.tensors, outfile);
        outfile.close(); // Ensures memory buffer flushes completely to disk!
        std::cout << "Successfully saved tensor summary to tensor_summary.txt\n";
    }
    else {
        std::cerr << "Error: Could not open output file for writing.\n";
    }
    int prefetch = 1;
    if(argc >= 4){
        prefetch = std::stoll(argv[3]);
    }
    std::cout << "Excess_time: " << estimate_linear(graph, prefetch) << "\n";
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}