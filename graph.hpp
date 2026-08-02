#pragma once

#include <iostream>
#include <string>
#include <variant>
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

    // Nodes and edges
    std::unordered_map<std::string, bool> isInit;
    std::unordered_map<std::string, tmd*> master_tensor_map;
    std::vector<tmd*> tensors;
    std::vector<onnx::NodeProto> nodes;

    void build_graph(const onnx::ModelProto& model);
};

// ============================================================================
// Helper & Utility Function Declarations
// ============================================================================

// Recursive symbolic dimension printer
void print_dim_expression(const Dim& dim, std::ostream& os = std::cout);

// Extract DataType from ONNX ValueInfoProto
DataType getDtype(const onnx::ValueInfoProto& value_info);

// Extract vector of Dim from ONNX ValueInfoProto
std::vector<Dim> getShape(onnx::ValueInfoProto value_info);

// Heuristic to estimate tensor layout based on rank
Layout getLayout(tmd* tensor);

// Extract shape dimensions from ONNX initializer (weights)
std::vector<Dim> get_initializer_shape(const onnx::TensorProto& initializer);

// Format and print all tensor metadata in graph
void print_tensor_vector_metadata(const std::vector<tmd*>& tensors, std::ostream& os = std::cout);

// Process uninferred ONNX ValueInfo inputs/outputs/intermediates
void tensor_uninfered(
    tmd* tensor, 
    onnx::ValueInfoProto tens, 
    bool isInput, 
    std::unordered_map<std::string, tmd*>& master_tensor_map, 
    std::vector<tmd*>& tensors
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