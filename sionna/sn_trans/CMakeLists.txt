file(GLOB_RECURSE SRCS *.cpp)
find_package(Torch REQUIRED)
add_library(sn_trans ${SRCS})   
add_subdirectory(sn_channel)

# so we can include "froze.h" in main.cpp
target_include_directories(
    sn_trans 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# note that becuase we want to include sionna/sn_utils in sn_channel , we need to link to sionn in here
target_link_libraries(
    sn_trans  
    torch_utils
    sionna
    ${TORCH_LIBRARIES}
)