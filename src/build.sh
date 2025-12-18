#!/bin/bash

# Default make command (single core) and NEP types
MAKE_CMD="make"
NEP_TYPES=20
COMPILE_FORTRAN=0

# Function to display help information
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -h              Show this help message"
    echo "  -jN             Use N parallel jobs for compilation (e.g., -j4)"
    echo "  -nN             Set number of NEP types (default: 20)"
    echo "  -m nn           Compile Fortran codes (required for NN and Linear models)"
    echo ""
    echo "Examples:"
    echo "  $0                     # Default compilation without Fortran"
    echo "  $0 -j4                 # Use 4 parallel jobs"
    echo "  $0 -m nn               # Compile Fortran codes"
    echo "  $0 -j4 -n50 -m nn      # Use 4 jobs, 50 NEP types, compile Fortran"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h)
            show_help
            ;;
        -j*)
            MAKE_CMD="make $1"
            shift
            ;;
        -n*)
            # Extract number after -n
            NEP_TYPES="${1#-n}"
            # If no number, check if next argument is numeric
            if [[ -z "$NEP_TYPES" ]]; then
                if [[ "$2" =~ ^[0-9]+$ ]]; then
                    NEP_TYPES="$2"
                    shift
                else
                    echo "Error: -n requires a numeric argument"
                    exit 1
                fi
            fi
            shift
            ;;
        -m)
            if [[ "$2" == "nn" ]]; then
                COMPILE_FORTRAN=1
                shift 2
            else
                echo "Error: -m option requires 'nn' argument"
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            ;;
    esac
done

echo "Using NEP_TYPES = $NEP_TYPES"
echo "Using MAKE_CMD = $MAKE_CMD"
echo "Compile Fortran codes: $([ $COMPILE_FORTRAN -eq 1 ] && echo "Yes" || echo "No")"

mkdir -p bin
mkdir -p lib

# Compile Fortran codes if requested
if [[ $COMPILE_FORTRAN -eq 1 ]]; then
    echo "Compiling Fortran codes..."
    
    # List of directories containing Fortran code
    FORTRAN_DIRS=("pre_data/gen_feature" "pre_data/fit" "pre_data/fortran_code" "md/fortran_code")
    
    for dir in "${FORTRAN_DIRS[@]}"; do
        echo "Compiling in $dir..."
        if ! make -C "$dir"; then
            echo "Error: Compilation failed in $dir"
            echo "Fortran compilation is required for NN and Linear models."
            exit 1
        fi
    done
    
    # Check for required Fortran compiled files
    REQUIRED_BIN_FILES=("main_MD.x" "gen_dR.x")
    MISSING_BIN_FILES=()
    
    for file in "${REQUIRED_BIN_FILES[@]}"; do
        if [[ ! -f "bin/$file" ]]; then
            MISSING_BIN_FILES+=("$file")
        fi
    done
    
    if [[ ${#MISSING_BIN_FILES[@]} -gt 0 ]]; then
        echo "Error: Missing required Fortran compiled files: ${MISSING_BIN_FILES[*]}"
        exit 1
    fi
    
    if [[ ! -f "lib/NeighConst.so" ]]; then
        echo "Error: lib/NeighConst.so not found (Fortran compilation product)"
        exit 1
    fi
    
    echo "Fortran compilation completed successfully"
else
    echo "Skipping Fortran compilation (NN and Linear models will not be available)"
fi

# make nep-cpu interface
echo "Building NEP-CPU interface..."
cd feature/nep_find_neigh
rm -rf build/*
rm -f findneigh.so 2>/dev/null
mkdir -p build
cd build
if cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && $MAKE_CMD; then
    cp findneigh.* ../findneigh.so 2>/dev/null
else
    echo "Warning: Failed to build NEP-CPU interface"
fi
cd ../../

# make nep-gpu interface
echo "Building NEP-GPU interface..."
mkdir -p NEP_GPU/build
cd NEP_GPU/build
if cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && $MAKE_CMD; then
    cp nep3_module*.so nep_gpu.so 2>/dev/null
else
    echo "Warning: Failed to build NEP-GPU interface"
fi
cd ../../../

# Build operators
echo "Building operators..."
cd op
rm -rf build
mkdir -p build
cd build
# for bigmodel the types should be 100
if cmake -DNEP_TYPES=$NEP_TYPES .. && $MAKE_CMD; then
    echo "Operators built successfully"
else
    echo "Warning: Failed to build operators"
fi
cd ..
cd ..

# Create symbolic links in bin directory
echo "Creating symbolic links in bin directory..."
cd bin

# Create symbolic link for MD executable only if it exists
if [[ -f "../md/fortran_code/main_MD.x" ]]; then
    ln -sf ../md/fortran_code/main_MD.x .
    echo "Created symbolic link for main_MD.x"
elif [[ $COMPILE_FORTRAN -eq 1 ]]; then
    echo "Error: main_MD.x should have been created by Fortran compilation but was not found"
    exit 1
else
    echo "Note: main_MD.x not available (requires Fortran compilation with -m nn)"
fi

# Create symbolic links for Python executables
ln -sf ../../main.py ./MATPL
ln -sf ../../main.py ./matpl
ln -sf ../../main.py ./MatPL
ln -sf ../../main.py ./PWMLFF
ln -sf ../../main.py ./pwmlff
# ln -sf ../../main_mnode.py ./MNEP

cd ..            # back to src dir

current_path=$(pwd)
parent_path=$(dirname "$current_path")

# write environment to env.sh
cat <<EOF > ../env.sh
# Environment for MatPL
export PYTHONPATH=$parent_path:\$PYTHONPATH
export PATH=$current_path/bin:\$PATH
EOF

echo ""
echo "================================="
if [[ $COMPILE_FORTRAN -eq 0 ]]; then
    echo "WARNING: Fortran codes were not compiled."
    echo "NN and Linear models will not be available."
    echo "To enable these models, recompile with the '-m nn' option:"
    echo "  sh build.sh -m nn"
    echo ""
fi

echo "MatPL has been successfully installed."
echo "Please load the MatPL environment variables before use."
echo ""
echo "Recommended method:"
echo "  source $parent_path/env.sh"
echo ""
echo "Or manually set environment variables:"
echo "  export PYTHONPATH=$parent_path:\$PYTHONPATH"
echo "  export PATH=$current_path/bin:\$PATH"
echo "=================================="
