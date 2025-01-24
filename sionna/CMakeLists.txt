file(GLOB_RECURSE SRCS *.cpp)
find_package(Torch REQUIRED)
add_library(sionna ${SRCS})   

add_subdirectory(sn_trans)
# so we can include "sim.h" in main.cpp
target_include_directories(
    sionna 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    sionna  
    torch_utils
    sys_model
    ${TORCH_LIBRARIES}
)