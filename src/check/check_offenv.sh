#!/bin/bash

# 相比于 在线版本多了多 openpmi 的检查
# 环境检查结果标志
ENV_PASS=true
FORTRAN_COMPATIBLE=true
CUDA_COMPATIBLE=true
OPENMPI_FOUND=true   # 新增 OpenMPI 标志

# 检查 ifort 编译器版本是否不小于19.1 和 MKL 库是否存在
check_ifort_mkl() {
    local ifort_ok=true
    local mkl_ok=true
    
    echo "=== Checking ifort compiler and MKL library ==="
    
    if command -v ifort &> /dev/null; then
        ifort_version=$(ifort --version | grep "ifort" | awk '{print $3}' | cut -d'.' -f1-2)
        major_version=$(echo $ifort_version | cut -d'.' -f1)
        minor_version=$(echo $ifort_version | cut -d'.' -f2)
        
        if [ "$major_version" -gt 19 ] || ([ "$major_version" -eq 19 ] && [ "$minor_version" -ge 1 ]); then
            echo "✓ ifort version: $ifort_version (>= 19.1)"
        else
            echo "✗ ifort version: $ifort_version (< 19.1)"
            ifort_ok=false
            FORTRAN_COMPATIBLE=false
            ENV_PASS=false
        fi
    else
        echo "✗ ifort compiler not found"
        ifort_ok=false
        FORTRAN_COMPATIBLE=false
        ENV_PASS=false
    fi
    
    # 检查 MKL 库是否存在
    if [ -d "/opt/intel/mkl" ] || [ -d "$MKLROOT" ] || [ ! -z "$MKLROOT" ]; then
        echo "✓ MKL library is installed"
    else
        echo "✗ MKL library is not installed"
        mkl_ok=false
        FORTRAN_COMPATIBLE=false
        ENV_PASS=false
    fi
}

# 检查 GCC 版本是否为 8.x 或更高
check_gcc_version() {
    echo "=== Checking GCC version ==="
    
    if command -v gcc &> /dev/null; then
        gcc_version=$(gcc -dumpversion | cut -d. -f1)
        if [ "$gcc_version" -ge 8 ]; then
            echo "✓ GCC version: $(gcc -dumpversion) (>= 8.0)"
        else
            echo "✗ GCC version: $(gcc -dumpversion) (< 8.0)"
            ENV_PASS=false
        fi
    else
        echo "✗ GCC not found"
        ENV_PASS=false
    fi
}

# 检查 CUDA 版本是否大于等于 11.8
check_cuda_version() {
    echo "=== Checking CUDA version ==="
    
    if command -v nvcc &> /dev/null; then
        cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d"," -f1 | cut -c2-)
        major_version=$(echo $cuda_version | cut -d'.' -f1)
        minor_version=$(echo $cuda_version | cut -d'.' -f2)

        if [ "$major_version" -gt 11 ] || ([ "$major_version" -eq 11 ] && [ "$minor_version" -ge 8 ]); then
            echo "✓ CUDA version: $cuda_version (>= 11.8)"
        else
            echo "✗ CUDA version: $cuda_version (< 11.8)"
            CUDA_COMPATIBLE=false
            ENV_PASS=false
        fi
    else
        echo "✗ nvcc command not found, CUDA might not be installed"
        CUDA_COMPATIBLE=false
        ENV_PASS=false
    fi
}

# 检查是否存在 nvcc 命令
check_nvcc() {
    echo "=== Checking nvcc availability ==="
    
    if command -v nvcc &> /dev/null; then
        echo "✓ nvcc command exists"
    else
        echo "✗ nvcc command does not exist"
        CUDA_COMPATIBLE=false
        ENV_PASS=false
    fi
}

# 新增：检查 OpenMPI（区分其他 MPI，并验证版本 ≥ 4.0）
check_openmpi() {
  echo "=== Checking OpenMPI (needed in MatPL-2026.3 lammps interface) ==="
  local ompi_found=false
  local ompi_version=""

  # 方法1: 检查 ompi_info 命令（OpenMPI 特有）
  if command -v ompi_info &> /dev/null; then
    ompi_found=true
    ompi_version=$(ompi_info --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    echo "✓ OpenMPI found (via ompi_info), version: ${ompi_version:-unknown}"
  # 方法2: 检查 mpirun --version 输出是否包含 "Open MPI"
  elif command -v mpirun &> /dev/null; then
    mpirun_version_output=$(mpirun --version 2>&1)
    if echo "$mpirun_version_output" | grep -qi "Open MPI"; then
      ompi_found=true
      ompi_version=$(echo "$mpirun_version_output" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
      echo "✓ OpenMPI found (via mpirun), version: ${ompi_version:-unknown}"
    else
      # 有其他 MPI 实现（如 Intel MPI），但不是 OpenMPI
      echo "⚠ Other MPI implementation detected (not OpenMPI):"
      echo "  $(mpirun --version 2>&1 | head -1)"
    fi
  fi

  if [ "$ompi_found" = true ]; then
    if [ -n "$ompi_version" ]; then
      major_version=$(echo "$ompi_version" | cut -d'.' -f1)
      if [ "$major_version" -ge 4 ]; then
        echo "✓ OpenMPI version meets requirement (>= 4.0)"
        OPENMPI_FOUND=true
      else
        echo "✗ OpenMPI version: $ompi_version (< 4.0)"
        OPENMPI_FOUND=false
      fi
    else
      # 无法获取版本，但 OpenMPI 存在，假设满足要求
      echo "⚠ OpenMPI detected but version could not be determined. Assuming compatibility."
      OPENMPI_FOUND=true
    fi
  else
    echo "✗ OpenMPI not found"
    OPENMPI_FOUND=false
  fi
}

# 执行检查
echo "========================================"
echo "      Environment Check Starting"
echo "========================================"
echo ""

check_ifort_mkl
echo ""
check_gcc_version
echo ""

# 检查 CUDA 相关信息 - 现在是强制要求
check_cuda_version
echo ""
check_nvcc
echo ""

# 新增 OpenMPI 检查
check_openmpi
echo ""

echo "========================================"
echo "        Environment Summary"
echo "========================================"

# 总结输出
if [ "$ENV_PASS" = true ]; then
    echo "✓ Environment check completed. All requirements are satisfied."
else
    echo "✗ Environment check failed. Please review the above errors."
fi

# Fortran 警告（始终显示，即使 ENV_PASS 为 true）
if [ "$FORTRAN_COMPATIBLE" = false ]; then
    echo ""
    echo "⚠ Warning: ifort compiler or MKL library is missing or does not meet version requirements."
    echo "  This will affect the compilation of Linear and NN models, but does not affect the use of DP and NEP models."
fi

# 新增 OpenMPI 英文警告
if [ "$OPENMPI_FOUND" = false ]; then
    echo ""
    echo "⚠ Warning: OpenMPI not detected or version below 4.0. This will affect the compilation of the MatPL-2026.3 lammps interface (we recommend version 4.x or higher)."
fi

echo "========================================"