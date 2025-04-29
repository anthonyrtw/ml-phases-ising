cmake_minimum_required(VERSION 3.10)

project(ML_phases_ising)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------ Path Settings ------------------
set(MAIN_PATH "${CMAKE_SOURCE_DIR}")
set(CMAKE_PREFIX_PATH "/path/to/libtorch")

# ------------------ Torch Settings ------------------
include_directories("/path/to/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}") 

add_executable(MY_MAIN_EXECUTABLE "${MAIN_PATH}/data_gen.cpp")
target_link_libraries(MY_MAIN_EXECUTABLE "${TORCH_LIBRARIES}")


# -------------------- REMOVE WINDOWS ERRORS --------------------
# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (WIN32)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET MY_MAIN_EXECUTABLE 
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:MY_MAIN_EXECUTABLE>)
endif (WIN32)
