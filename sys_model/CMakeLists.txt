file(GLOB_RECURSE SRCS *.cpp)
find_package(Torch REQUIRED)
#create library "sys_model"
add_library(sys_model ${SRCS})   

# so we can include "froze.h" in main.cpp
target_include_directories(
    sys_model 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#link this model to polar lib & torch.h  
# sn_channel is for awgn.h   
target_link_libraries(
    sys_model  
    polar
    sn_trans
    sn_channel
    ${TORCH_LIBRARIES}
)