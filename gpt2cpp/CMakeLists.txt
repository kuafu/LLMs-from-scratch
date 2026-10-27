cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gpt_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA support" OFF)

# 优先查找用户指定的路径或默认路径 (H:/LLM/libtorch 或 F:/LLM/libtorch)
if(NOT DEFINED CMAKE_PREFIX_PATH)
    if(EXISTS "H:/LLM/libtorch")
        set(CMAKE_PREFIX_PATH "H:/LLM/libtorch")
    elseif(EXISTS "F:/LLM/libtorch")
        set(CMAKE_PREFIX_PATH "F:/LLM/libtorch")
    endif()
endif()

message(STATUS "Using LibTorch from: ${CMAKE_PREFIX_PATH}")

# 禁用 CUDA 或在未安装 CUDA Toolkit 时，避免 CUDA 版 LibTorch 强制探测 nvcc / CUDA Toolkit
if(NOT USE_CUDA)
    message(STATUS "CUDA disabled: running in CPU mode.")
    if(NOT TARGET torch::cudart)
        add_library(torch::cudart INTERFACE IMPORTED)
    endif()
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(gpt gpt.cpp)
target_link_libraries(gpt "${TORCH_LIBRARIES}")

set_property(TARGET gpt PROPERTY CXX_STANDARD 20)

if (MSVC)
    target_compile_options(gpt PRIVATE /utf-8 /bigobj /wd4267)
    
    # 自动将 libtorch 中的 DLL 拷贝到生成目录下，方便直接运行可执行文件
    add_custom_command(TARGET gpt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${TORCH_INSTALL_PREFIX}/lib"
        "$<TARGET_FILE_DIR:gpt>"
    )
endif()
