cmake_minimum_required(VERSION 3.10)
project(ONNXParser CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)
find_package(absl QUIET)

add_library(onnx_proto SHARED onnx.pb.cc)
add_executable(onnx_parser test_parser.cpp)

add_library(shape_infer SHARED shapeInfer.cpp)
target_link_libraries(shape_infer PUBLIC onnx_proto)

target_link_libraries(onnx_proto PUBLIC protobuf::libprotobuf)

if(UNIX AND NOT APPLE)
    target_link_libraries(onnx_proto PUBLIC absl_log_internal_check_op absl_log_internal_message)
endif()

add_library(graph SHARED graph.cpp)
target_link_libraries(graph PUBLIC shape_infer)

target_link_libraries(onnx_parser PRIVATE graph)
