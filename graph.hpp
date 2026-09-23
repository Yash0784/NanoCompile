#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>

#include "onnx.pb.h" 
#include "shapeInfer.hpp"

// ============================================================================
// ONNX Graph Class
// ============================================================================

class ONNXGraph {
public:
    // Model Metadata
    int64_t ir_version = 0;
    int64_t model_version = 0;
    std::string producer_name = "Unknown";
    std::string producer_version = "Unknown";
    std::string domain = "";

    // The concrete batch size this graph was resolved for. Every symbolic
    // batch axis in the ONNX file gets substituted with this value during
    // build_graph, so the whole graph (and every tensor's bytes/num_elements)
    // is only valid for this one batch size. Re-run build_graph with a
    // different value to plan for a different batch size.
    int64_t batch_size = 1;

    // Nodes and edges
    std::unordered_map<std::string, bool> isInit;
    std::unordered_map<std::string, tmd*> master_tensor_map;
    std::vector<tmd*> tensors;
    std::vector<onnx::NodeProto> nodes;
    std::vector<float> nodeTime;
    std::unordered_map<int, int> layer;

    void build_graph(const onnx::ModelProto& model, int64_t batch_size = 1);
};

// ============================================================================
// Helper & Utility Function Declarations
// ============================================================================

// Extract DataType from ONNX ValueInfoProto
DataType getDtype(const onnx::ValueInfoProto& value_info);

// Extract a fully concrete shape (vector<int64_t>) from an ONNX
// ValueInfoProto. Any symbolic axis (dim_param) is substituted with
// `batch_size` -- see the long comment on the .cpp definition for why this
// project treats every symbolic dim_param as the batch axis.
std::vector<int64_t> getShape(onnx::ValueInfoProto value_info, int64_t batch_size);

// Heuristic to estimate tensor layout based on rank
Layout getLayout(tmd* tensor);

// Extract shape dimensions from ONNX initializer (weights). Initializer
// dims are always concrete in ONNX (no symbolic axis is ever legal there),
// so this needs no batch_size argument.
std::vector<int64_t> get_initializer_shape(const onnx::TensorProto& initializer);

// Format and print all tensor metadata in graph
void print_tensor_vector_metadata(const std::vector<tmd*>& tensors, std::ostream& os);

// Process uninferred ONNX ValueInfo inputs/outputs/intermediates
void tensor_uninfered(
    tmd* tensor, 
    onnx::ValueInfoProto tens, 
    bool isInput, 
    std::unordered_map<std::string, tmd*>& master_tensor_map, 
    std::vector<tmd*>& tensors,
    int64_t batch_size
);

// Graph analysis pass: Link producer and consumer node pointers to tensors
void link_tensor_producers_and_consumers(
    const std::vector<onnx::NodeProto>& topo_nodes,
    std::unordered_map<std::string, tmd*>& master_tensor_map
);

// Graph analysis pass: Compute tensor lifetime ranges [born_at, first_use, last_use]
void life_time_tens(
    const std::vector<onnx::NodeProto>& TopoNodes, 
    std::unordered_map<std::string, tmd*>& master_tensor_map
);
