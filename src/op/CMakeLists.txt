add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# 1. Basic info
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(CalcOps)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# 2. Include Torch and Python paths
execute_process(COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH)
string(REPLACE "\n" "" TORCH_CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR)
string(REPLACE "\n" "" PYTHON_INCLUDE_DIR ${PYTHON_INCLUDE_DIR})
execute_process(COMMAND python3 -c "import torch; print(torch.__path__[0])" OUTPUT_VARIABLE TORCH_PATH)
string(REPLACE "\n" "" TORCH_PATH ${TORCH_PATH})

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${TORCH_PATH}/include/)
include_directories(${TORCH_PATH}/include/torch/csrc/api/include/)

# 3. Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

# 4. Check CUDA availability - both pytorch support and system toolkit
# 4.1 Check if PyTorch has CUDA support
execute_process(
    COMMAND python3 -c "import torch; print(torch.cuda.is_available())"
    OUTPUT_VARIABLE PYTORCH_HAS_CUDA
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# 4.2 Check if CUDA toolkit is installed on the system
find_program(NVCC nvcc)
if(NVCC)
    set(SYSTEM_HAS_CUDA TRUE)
    # Extract CUDA version
    execute_process(
        COMMAND ${NVCC} --version
        OUTPUT_VARIABLE NVCC_VERSION_OUTPUT
    )
    # Parse version
    string(REGEX MATCH "V[0-9]+\\.[0-9]+\\.[0-9]+" CUDA_VERSION ${NVCC_VERSION_OUTPUT})
    string(REPLACE "V" "" CUDA_VERSION ${CUDA_VERSION})
    message(STATUS "Found CUDA toolkit version: ${CUDA_VERSION}")
else()
    set(SYSTEM_HAS_CUDA FALSE)
    message(STATUS "CUDA toolkit (nvcc) not found in system PATH")
endif()

# 4.3 Check CUDA_HOME or CUDA_PATH environment variables as fallback
if(NOT SYSTEM_HAS_CUDA)
    if(DEFINED ENV{CUDA_HOME} AND EXISTS $ENV{CUDA_HOME})
        set(SYSTEM_HAS_CUDA TRUE)
        message(STATUS "Found CUDA_HOME: $ENV{CUDA_HOME}")
    elseif(DEFINED ENV{CUDA_PATH} AND EXISTS $ENV{CUDA_PATH})
        set(SYSTEM_HAS_CUDA TRUE)
        message(STATUS "Found CUDA_PATH: $ENV{CUDA_PATH}")
    endif()
endif()

# 5. Determine if GPU compilation should proceed
if(PYTORCH_HAS_CUDA AND SYSTEM_HAS_CUDA)
    set(COMPILE_GPU TRUE)
    message(STATUS "PyTorch has CUDA support AND CUDA toolkit is available. Enabling GPU compilation.")
elseif(PYTORCH_HAS_CUDA AND NOT SYSTEM_HAS_CUDA)
    set(COMPILE_GPU FALSE)
    message(WARNING "PyTorch was built with CUDA support but CUDA toolkit is not available.")
    message(WARNING "Please install CUDA toolkit or set CUDA_HOME/CUDA_PATH environment variable.")
    message(WARNING "Skipping GPU compilation. Only CPU operators will be built.")
elseif(NOT PYTORCH_HAS_CUDA AND SYSTEM_HAS_CUDA)
    set(COMPILE_GPU FALSE)
    message(WARNING "CUDA toolkit is available but PyTorch was not built with CUDA support.")
    message(WARNING "Please install PyTorch with CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX")
    message(WARNING "Skipping GPU compilation. Only CPU operators will be built.")
else()
    set(COMPILE_GPU FALSE)
    message(STATUS "No CUDA support detected. Building only CPU operators.")
endif()

# 6. Always add CPU subdirectory
add_subdirectory(cmake/cpu)

# 7. Conditionally add GPU subdirectory
if(COMPILE_GPU)
    message(STATUS "Adding GPU subdirectory for CUDA compilation.")
    add_subdirectory(cmake/gpu)
else()
    message(STATUS "Skipping GPU subdirectory.")
endif()
