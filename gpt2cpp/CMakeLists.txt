cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gpt_cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA support" OFF)

# Use the shell's Python unless the caller explicitly selected an interpreter.
# The same interpreter is used to discover PyTorch and the embedding library.
if(NOT DEFINED Python3_EXECUTABLE OR "${Python3_EXECUTABLE}" STREQUAL "")
    unset(_GPT_PYTHON_ON_PATH CACHE)
    find_program(_GPT_PYTHON_ON_PATH NAMES python python3 NAMES_PER_DIR
        PATHS ENV PATH NO_DEFAULT_PATH)
    if(NOT _GPT_PYTHON_ON_PATH)
        message(FATAL_ERROR
            "Python was not found on PATH. Activate the intended environment or set Python3_EXECUTABLE.")
    endif()
    set(Python3_EXECUTABLE "${_GPT_PYTHON_ON_PATH}" CACHE FILEPATH
        "Python interpreter used for PyTorch discovery and embedding" FORCE)
    unset(_GPT_PYTHON_ON_PATH CACHE)
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Using Python3: ${Python3_EXECUTABLE}")

# Record the selected interpreter's environment for embedded Python.
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -X utf8 -c "import sys; print(sys.prefix)"
    RESULT_VARIABLE _gpt_python_prefix_result
    OUTPUT_VARIABLE _gpt_python_prefix
    ERROR_VARIABLE _gpt_python_prefix_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ENCODING UTF-8)
if(NOT "${_gpt_python_prefix_result}" STREQUAL "0" OR NOT IS_DIRECTORY "${_gpt_python_prefix}")
    message(FATAL_ERROR
        "Cannot determine the Python environment for ${Python3_EXECUTABLE}.\n${_gpt_python_prefix_error}")
endif()
message(STATUS "Python environment prefix: ${_gpt_python_prefix}")

# Changing the interpreter or Torch search hints in an existing build can
# leave cached Caffe2 and library paths pointing at a different distribution.
set(_gpt_prefix_hint "${CMAKE_PREFIX_PATH}")
set(_gpt_env_prefix_hint "$ENV{CMAKE_PREFIX_PATH}")
if(DEFINED GPT_CONFIGURED_PYTHON AND
   NOT "${GPT_CONFIGURED_PYTHON}" STREQUAL "${Python3_EXECUTABLE}")
    message(FATAL_ERROR "Python changed since this build was configured. Use a new build directory.")
endif()
if(DEFINED GPT_CONFIGURED_PYTHON_PREFIX AND
   NOT "${GPT_CONFIGURED_PYTHON_PREFIX}" STREQUAL "${_gpt_python_prefix}")
    message(FATAL_ERROR "Python's environment prefix changed. Use a new build directory.")
endif()
if(DEFINED GPT_CONFIGURED_PREFIX_HINT AND
   NOT "${GPT_CONFIGURED_PREFIX_HINT}" STREQUAL "${_gpt_prefix_hint}")
    message(FATAL_ERROR "CMAKE_PREFIX_PATH changed since this build was configured. Use a new build directory.")
endif()
if(DEFINED GPT_CONFIGURED_ENV_PREFIX_HINT AND
   NOT "${GPT_CONFIGURED_ENV_PREFIX_HINT}" STREQUAL "${_gpt_env_prefix_hint}")
    message(FATAL_ERROR "The CMAKE_PREFIX_PATH environment changed since this build was configured. Use a new build directory.")
endif()
if(DEFINED GPT_CONFIGURED_TORCH_DIR AND DEFINED Torch_DIR AND
   NOT "${GPT_CONFIGURED_TORCH_DIR}" STREQUAL "${Torch_DIR}")
    message(FATAL_ERROR "Torch_DIR changed since this build was configured. Use a new build directory.")
endif()

# A caller-supplied Torch_DIR or CMAKE_PREFIX_PATH selects a standalone package.
# Torch_DIR cached by our own previous Python discovery is not an override.
set(_gpt_manual_torch FALSE)
if(NOT "${_gpt_prefix_hint}" STREQUAL "" OR NOT "${_gpt_env_prefix_hint}" STREQUAL "")
    set(_gpt_manual_torch TRUE)
elseif(DEFINED Torch_DIR AND NOT "${Torch_DIR}" STREQUAL "" AND
       NOT "${Torch_DIR}" MATCHES "-NOTFOUND$")
    if(NOT DEFINED GPT_CONFIGURED_TORCH_SOURCE OR
       GPT_CONFIGURED_TORCH_SOURCE STREQUAL "manual")
        set(_gpt_manual_torch TRUE)
    endif()
endif()

# 禁用 CUDA 或在未安装 CUDA Toolkit 时，避免 CUDA 版 LibTorch 强制探测 nvcc / CUDA Toolkit
if(NOT USE_CUDA)
    message(STATUS "CUDA disabled: running in CPU mode.")
    if(NOT TARGET torch::cudart)
        add_library(torch::cudart INTERFACE IMPORTED)
    endif()
endif()

if(_gpt_manual_torch)
    if(DEFINED Torch_DIR AND NOT "${Torch_DIR}" STREQUAL "" AND
       NOT "${Torch_DIR}" MATCHES "-NOTFOUND$")
        if(NOT EXISTS "${Torch_DIR}/TorchConfig.cmake")
            message(FATAL_ERROR "Torch_DIR does not contain TorchConfig.cmake: ${Torch_DIR}")
        endif()
        find_package(Torch CONFIG REQUIRED PATHS "${Torch_DIR}" NO_DEFAULT_PATH)
    else()
        set(_gpt_manual_prefixes ${CMAKE_PREFIX_PATH})
        if(NOT "${_gpt_env_prefix_hint}" STREQUAL "")
            file(TO_CMAKE_PATH "${_gpt_env_prefix_hint}" _gpt_env_prefixes)
            list(APPEND _gpt_manual_prefixes ${_gpt_env_prefixes})
        endif()
        find_package(Torch CONFIG REQUIRED PATHS ${_gpt_manual_prefixes} NO_DEFAULT_PATH)
    endif()
    set(_gpt_torch_source "manual")
    if(EXISTS "${TORCH_INSTALL_PREFIX}/build-version")
        file(STRINGS "${TORCH_INSTALL_PREFIX}/build-version" _gpt_torch_version LIMIT_COUNT 1)
    else()
        set(_gpt_torch_version "${Torch_VERSION}")
    endif()
else()
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -X utf8 -c
            "import torch; print(torch.utils.cmake_prefix_path); print(torch.__version__)"
        RESULT_VARIABLE _gpt_torch_result
        OUTPUT_VARIABLE _gpt_torch_info
        ERROR_VARIABLE _gpt_torch_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ENCODING UTF-8)
    if(NOT "${_gpt_torch_result}" STREQUAL "0")
        message(FATAL_ERROR
            "Cannot import PyTorch using ${Python3_EXECUTABLE}. Install torch in that environment, "
            "select another Python3_EXECUTABLE, or specify a LibTorch CMAKE_PREFIX_PATH or Torch_DIR.\n${_gpt_torch_error}")
    endif()
    string(REPLACE "\r\n" "\n" _gpt_torch_info "${_gpt_torch_info}")
    string(REPLACE "\n" ";" _gpt_torch_info "${_gpt_torch_info}")
    list(LENGTH _gpt_torch_info _gpt_torch_info_count)
    if(NOT _gpt_torch_info_count EQUAL 2)
        message(FATAL_ERROR "Unexpected PyTorch discovery output from ${Python3_EXECUTABLE}: ${_gpt_torch_info}")
    endif()
    list(GET _gpt_torch_info 0 _gpt_python_torch_prefix)
    list(GET _gpt_torch_info 1 _gpt_torch_version)
    if(NOT EXISTS "${_gpt_python_torch_prefix}/Torch/TorchConfig.cmake")
        message(FATAL_ERROR "Python PyTorch has no TorchConfig.cmake under ${_gpt_python_torch_prefix}")
    endif()
    if(DEFINED GPT_CONFIGURED_PYTORCH_PREFIX AND
       NOT "${GPT_CONFIGURED_PYTORCH_PREFIX}" STREQUAL "${_gpt_python_torch_prefix}")
        message(FATAL_ERROR "Python's PyTorch location changed. Use a new build directory.")
    endif()
    if(DEFINED GPT_CONFIGURED_PYTORCH_VERSION AND
       NOT "${GPT_CONFIGURED_PYTORCH_VERSION}" STREQUAL "${_gpt_torch_version}")
        message(FATAL_ERROR "Python's PyTorch version changed. Use a new build directory.")
    endif()
    find_package(Torch CONFIG REQUIRED PATHS "${_gpt_python_torch_prefix}" NO_DEFAULT_PATH)
    get_filename_component(_gpt_expected_torch_dir "${_gpt_python_torch_prefix}/Torch" REALPATH)
    get_filename_component(_gpt_found_torch_dir "${Torch_DIR}" REALPATH)
    if(WIN32)
        string(TOLOWER "${_gpt_expected_torch_dir}" _gpt_expected_torch_dir)
        string(TOLOWER "${_gpt_found_torch_dir}" _gpt_found_torch_dir)
    endif()
    if(NOT "${_gpt_expected_torch_dir}" STREQUAL "${_gpt_found_torch_dir}")
        message(FATAL_ERROR "Torch_DIR does not match the selected Python's PyTorch. Use a new build directory.")
    endif()
    set(_gpt_torch_source "Python")
    set(GPT_CONFIGURED_PYTORCH_PREFIX "${_gpt_python_torch_prefix}" CACHE INTERNAL "Python PyTorch prefix")
    set(GPT_CONFIGURED_PYTORCH_VERSION "${_gpt_torch_version}" CACHE INTERNAL "Python PyTorch version")
endif()

# TorchConfig can be redirected by TORCH_INSTALL_PREFIX or stale dependency
# cache entries. Reject mixed distributions before linking their libraries.
get_filename_component(_gpt_config_root "${Torch_DIR}/../../.." REALPATH)
get_filename_component(_gpt_install_root "${TORCH_INSTALL_PREFIX}" REALPATH)
get_filename_component(_gpt_expected_caffe2 "${Torch_DIR}/../Caffe2" REALPATH)
if(DEFINED Caffe2_DIR AND NOT "${Caffe2_DIR}" MATCHES "-NOTFOUND$")
    get_filename_component(_gpt_found_caffe2 "${Caffe2_DIR}" REALPATH)
endif()
if(WIN32)
    string(TOLOWER "${_gpt_config_root}" _gpt_config_root)
    string(TOLOWER "${_gpt_install_root}" _gpt_install_root)
    string(TOLOWER "${_gpt_expected_caffe2}" _gpt_expected_caffe2)
    string(TOLOWER "${_gpt_found_caffe2}" _gpt_found_caffe2)
endif()
if(NOT "${_gpt_config_root}" STREQUAL "${_gpt_install_root}" OR
   (DEFINED Caffe2_DIR AND NOT "${Caffe2_DIR}" MATCHES "-NOTFOUND$" AND
    NOT "${_gpt_expected_caffe2}" STREQUAL "${_gpt_found_caffe2}"))
    message(FATAL_ERROR "Torch and Caffe2 paths point to different installations. Use a new build directory and check TORCH_INSTALL_PREFIX.")
endif()
get_filename_component(_gpt_expected_lib_dir "${TORCH_INSTALL_PREFIX}/lib" REALPATH)
if(WIN32)
    string(TOLOWER "${_gpt_expected_lib_dir}" _gpt_expected_lib_dir)
endif()
foreach(_gpt_library IN ITEMS TORCH_LIBRARY c10_LIBRARY kineto_LIBRARY
                              CAFFE2_NVRTC_LIBRARY C10_CUDA_LIBRARY)
    if(DEFINED ${_gpt_library} AND NOT "${${_gpt_library}}" STREQUAL "" AND
       NOT "${${_gpt_library}}" MATCHES "-NOTFOUND$")
        if(NOT EXISTS "${${_gpt_library}}")
            message(FATAL_ERROR "${_gpt_library} does not exist: ${${_gpt_library}}")
        endif()
        get_filename_component(_gpt_library_path "${${_gpt_library}}" REALPATH)
        get_filename_component(_gpt_library_dir "${_gpt_library_path}" DIRECTORY)
        if(WIN32)
            string(TOLOWER "${_gpt_library_dir}" _gpt_library_dir)
        endif()
        if(NOT "${_gpt_library_dir}" STREQUAL "${_gpt_expected_lib_dir}")
            message(FATAL_ERROR
                "${_gpt_library} points outside the selected Torch installation: ${${_gpt_library}}. "
                "Use a matching library in ${TORCH_INSTALL_PREFIX}/lib and a new build directory.")
        endif()
    endif()
endforeach()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "Torch source: ${_gpt_torch_source}")
message(STATUS "Torch version: ${_gpt_torch_version}")
message(STATUS "Torch_DIR: ${Torch_DIR}")
message(STATUS "Torch install prefix: ${TORCH_INSTALL_PREFIX}")

set(GPT_CONFIGURED_PYTHON "${Python3_EXECUTABLE}" CACHE INTERNAL "Configured Python")
set(GPT_CONFIGURED_PYTHON_PREFIX "${_gpt_python_prefix}" CACHE INTERNAL "Configured Python environment")
set(GPT_CONFIGURED_PREFIX_HINT "${_gpt_prefix_hint}" CACHE INTERNAL "Configured CMake prefix hint")
set(GPT_CONFIGURED_ENV_PREFIX_HINT "${_gpt_env_prefix_hint}" CACHE INTERNAL "Configured environment prefix hint")
set(GPT_CONFIGURED_TORCH_DIR "${Torch_DIR}" CACHE INTERNAL "Configured Torch directory")
set(GPT_CONFIGURED_TORCH_SOURCE "${_gpt_torch_source}" CACHE INTERNAL "Configured Torch source")

set(GPT_PYTHON_EXECUTABLE_LITERAL "${Python3_EXECUTABLE}")
set(GPT_PYTHON_PREFIX_LITERAL "${_gpt_python_prefix}")
foreach(_gpt_literal IN ITEMS GPT_PYTHON_EXECUTABLE_LITERAL GPT_PYTHON_PREFIX_LITERAL)
    string(REPLACE "\\" "\\\\" ${_gpt_literal} "${${_gpt_literal}}")
    string(REPLACE "\"" "\\\"" ${_gpt_literal} "${${_gpt_literal}}")
endforeach()
configure_file(PythonEnvironment.h.in "${CMAKE_CURRENT_BINARY_DIR}/generated/PythonEnvironment.h" @ONLY)

add_executable(gpt gpt.cpp TiktokenDecoder.cpp TiktokenDecoder.h)
target_include_directories(gpt PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(gpt PRIVATE "${TORCH_LIBRARIES}" Python3::Python)

set_property(TARGET gpt PROPERTY CXX_STANDARD 20)

if (MSVC)
    # Visual Studio loads this from the project and embeds it in generated PDBs.
    target_sources(gpt PRIVATE debug/libtorch.natvis)
    source_group("Debugger" FILES debug/libtorch.natvis)
    target_compile_options(gpt PRIVATE /utf-8 /bigobj /wd4267)
    
    # 自动将 libtorch 中的 DLL 拷贝到生成目录下，方便直接运行可执行文件
    add_custom_command(TARGET gpt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${TORCH_INSTALL_PREFIX}/lib"
        "$<TARGET_FILE_DIR:gpt>"
    )
endif()
