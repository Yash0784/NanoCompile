#include "swap_plan.hpp"

float estimate_linear(ONNXGraph& model, int prefetch){
    float total_time = 0;
    for(tmd* tensor : model.tensors){
        if(tensor->is_initializer)
            total_time += tensor->swap_in_time;
    }
    float prefetch_time = 0;
    float compute_time = 0;
    for(int i = 0; i < prefetch; i++){
        compute_time += model.nodeTime[i];
        auto& node = model.nodes[i];
        for (const auto& in_name : node.input()) {
            if((model.master_tensor_map[in_name])->is_initializer)
                prefetch_time += (model.master_tensor_map[in_name])->swap_in_time;
        }
    }
    float excess_time = prefetch_time - total_time;

    for(int i = prefetch; i < model.nodes.size(); i++){
        auto& node = model.nodes[i];
        float fetch_time = 0;
        for (const auto& in_name : node.input()) {
            if((model.master_tensor_map[in_name])->is_initializer)
                fetch_time += (model.master_tensor_map[in_name])->swap_in_time;
        }
        compute_time = compute_time - model.nodeTime[i - prefetch];
        if(fetch_time - compute_time > 0){
            std::cout << "Excess time induce at node " << i << "\n";
            excess_time += fetch_time - compute_time;
            compute_time = model.nodeTime[i];
            int j = 1;
            for(j = 1; j + i < model.nodes.size() && j < prefetch; j++){
                compute_time += model.nodeTime[j + i];
                node = model.nodes[i + j];
                for (const auto& in_name : node.input()) {
                    if((model.master_tensor_map[in_name])->is_initializer)
                        excess_time += (model.master_tensor_map[in_name])->swap_in_time;
                }
            }
            i = i + j - 1;
            continue;
        }
        compute_time += model.nodeTime[i];
    }
    return excess_time;
}