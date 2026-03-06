#!/bin/bash

# Default make command (single core) and NEP types
MAKE_CMD="make"
PATCH_DIR=$(pwd)
BASE_DIR=$1
VERSION=$2
MATPL_DIR=${BASE_DIR}/$3
CLEAN_ALL=$4
CPU_ONLY=$5

# 设置环境目录
if [ "$CPU_ONLY" = "0" ]; then
  ENV_DIR=${BASE_DIR}/matpl-${VERSION}
else
  ENV_DIR=${BASE_DIR}/matpl_cpu-${VERSION}
fi

# 检查环境目录是否存在
if [ -f "$ENV_DIR/bin/activate" ]; then
  . "$ENV_DIR/bin/activate"
else
  echo "Warning: Virtual environment not found at $ENV_DIR/bin/activate"
fi

echo "patch file dir is $PATCH_DIR"
echo "MatPL root dir is $BASE_DIR"
echo "ENV   root dir is $ENV_DIR"

# 检查目录是否存在
if [ -d "$BASE_DIR" ]; then
  ls "$BASE_DIR"
else
  echo "Error: Base directory $BASE_DIR does not exist"
  exit 1
fi

cd "$MATPL_DIR/src" || {
  echo "Error: Cannot change to directory $MATPL_DIR/src"
  exit 1
}

# Parse command line arguments
# 注意：这里移除了位置参数 $1-$5，因为它们已经被使用
# 需要从 $6 开始处理额外的选项参数
shift 5  # 移除前5个位置参数

while [ $# -gt 0 ]; do
    case $1 in
        -j*)
            MAKE_CMD="make $1"
            shift
            ;;
        *)
            # 忽略其他参数
            shift
            ;;
    esac
done

echo "Using MAKE_CMD = $MAKE_CMD"

# 检查目录是否存在
if [ ! -d "op" ]; then
    echo "Error: 'op' directory not found in $(pwd)"
    exit 1
fi

cd op || {
    echo "Error: Cannot change to 'op' directory"
    exit 1
}

if [ "$CLEAN_ALL" = "1" ]; then
    rm -rf build
fi

# python setup.py install --user
mkdir -p build
cd build || {
    echo "Error: Cannot change to 'build' directory"
    exit 1
}

# for bigmodel the types should be 100
cmake ..
$MAKE_CMD

cd "$PATCH_DIR" || {
    echo "Error: Cannot return to patch directory $PATCH_DIR"
    exit 1
}

echo "OP operator compilation successful!"
