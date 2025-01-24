file(GLOB_RECURSE SRCS *.cpp)
find_package(Torch REQUIRED)
#create library "torch_utils"
add_library(torch_utils ${SRCS})   

# so we can include "torch_utils.h" in main.cpp
target_include_directories(
    torch_utils 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#we want use torch.h  
target_link_libraries(
    torch_utils  
    ${TORCH_LIBRARIES}
)