file(GLOB_RECURSE SRCS *.cpp)
find_package(Torch REQUIRED)
add_library(sn_channel ${SRCS}) 

# so we can include "froze.h" in main.cpp
target_include_directories(
    sn_channel
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    sn_channel
    sionna
    ${TORCH_LIBRARIES}
)