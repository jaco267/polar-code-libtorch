file(GLOB_RECURSE SRCS *.cpp)
find_package(Torch REQUIRED)
#create library "polar"
add_library(polar ${SRCS})   

# so we can include "froze.h" in main.cpp
target_include_directories(
    polar 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


target_link_libraries(
    polar  
    torch_utils
    ${TORCH_LIBRARIES}
)