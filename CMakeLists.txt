# CMakeLists.txt  #sudo update-alternatives --config c++ #change from g++ to clang
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project("main")
add_subdirectory(polar)
add_subdirectory(torch_utils)
add_subdirectory(sys_model)
add_subdirectory(sionna)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Collect all .cpp files in the project directory (not include sub dir)
file(GLOB SOURCES "*.cpp")

# add_executable(${PROJECT_NAME} main.cpp)
# Add the executable with all collected source files
add_executable(${PROJECT_NAME} ${SOURCES})

# add lib "polar"  (see polar/cmakelist.txt)
target_link_libraries(
    ${PROJECT_NAME} 
    polar 
    sionna
    sn_channel
    torch_utils   
    sys_model
    "${TORCH_LIBRARIES}")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
