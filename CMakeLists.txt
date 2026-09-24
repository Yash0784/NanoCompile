cmake_minimum_required(VERSION 3.18)
project(ONNXParser CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA support
enable_language(CUDA)

# Find standard dependencies
find_package(Protobuf REQUIRED)
find_package(absl QUIET)
find_package(CUDAToolkit REQUIRED)

# Manually find cuDNN library (Arch places cudnn in /usr/lib or /opt/cuda/lib)
find_library(CUDNN_LIBRARY NAMES cudnn HINTS /usr/lib /opt/cuda/lib /usr/local/cuda/lib64 REQUIRED)
find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h HINTS /usr/include /opt/cuda/include /usr/local/cuda/include REQUIRED)

# Build libraries
add_library(onnx_proto SHARED onnx.pb.cc)
add_library(shape_infer SHARED shapeInfer.cpp)
add_library(graph SHARED graph.cpp)
add_library(swap_plan SHARED swap_plan.cpp)

# Create the op_profiler library
add_library(op_profiler SHARED op_profiler.cpp)
target_include_directories(op_profiler PRIVATE ${CUDNN_INCLUDE_DIR})

# Link dependencies to onnx_proto
target_link_libraries(onnx_proto PUBLIC protobuf::libprotobuf)

if(UNIX AND NOT APPLE)
    target_link_libraries(onnx_proto PUBLIC absl_log_internal_check_op absl_log_internal_message)
endif()

# Link dependencies to shape_infer, graph, and swap_plan
target_link_libraries(shape_infer PUBLIC onnx_proto)
target_link_libraries(graph PUBLIC shape_infer)
target_link_libraries(swap_plan PUBLIC graph)

# Link CUDA, cuBLAS, cuDNN, and graph libraries to op_profiler
target_link_libraries(op_profiler PUBLIC 
    graph
    CUDA::cudart
    CUDA::cublas
    ${CUDNN_LIBRARY}
)

# Build main executable and link op_profiler, graph, and swap_plan
add_executable(onnx_parser test_parser.cpp)
target_link_libraries(onnx_parser PRIVATE graph op_profiler swap_plan)