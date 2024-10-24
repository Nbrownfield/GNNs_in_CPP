cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(GCN)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(GCN main.cpp)
target_link_libraries(GCN "${TORCH_LIBRARIES}")

set_property(TARGET GCN PROPERTY CXX_STANDARD 17)