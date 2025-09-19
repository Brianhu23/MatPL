#!/bin/bash

# 获取当前日期（年月日格式，带前导零）
CURRENT_DATE=$(date +%Y.%m.%d)

# 定义文件名
SOURCE_DIR="Patch-MatPL-2025.3"
PWACT_DIR="pwact-0.4.2.tar.gz"
PWDATA_DIR="pwdata-0.5.4.tar.gz"
TAR_FILE="matpl-patch-${CURRENT_DATE}.tar"
GZ_FILE="${TAR_FILE}.gz"
BASE64_FILE="${GZ_FILE}.base64"
SH_FILE="matpl-patch-${CURRENT_DATE}.sh"
FINAL_SH_TAR="matpl-patch-${CURRENT_DATE}.sh.tar.gz"
TEMPLATE_FILE="patch-matpl.template"
CHECK_SCRIPT="check_offenv.sh"

# 清理标志，默认不清理
CLEANUP_FLAG=false

# 颜色输出函数
red() { echo -e "\033[31m$1\033[0m"; }
green() { echo -e "\033[32m$1\033[0m"; }
yellow() { echo -e "\033[33m$1\033[0m"; }
blue() { echo -e "\033[34m$1\033[0m"; }

# 显示使用帮助
show_usage() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  -c, --clean     打包完成后自动清理临时文件"
    echo "  -h, --help      显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0              # 打包但不清理临时文件"
    echo "  $0 -c           # 打包并清理临时文件"
    echo "  $0 --clean      # 打包并清理临时文件"
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--clean)
                CLEANUP_FLAG=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                red "未知选项: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# 检查必要文件是否存在
check_files() {
    if [ ! -d "$SOURCE_DIR" ]; then
        red "错误: 目录 $SOURCE_DIR 不存在!"
        exit 1
    fi
    
    if [ ! -f "$TEMPLATE_FILE" ]; then
        red "错误: 模板文件 $TEMPLATE_FILE 不存在!"
        exit 1
    fi
    
    if [ ! -f "$CHECK_SCRIPT" ]; then
        yellow "警告: 检查脚本 $CHECK_SCRIPT 不存在，将继续执行但不包含该文件"
    fi
}

# 第一步：打包并压缩
step1_package() {
    green "步骤1: 打包目录 $SOURCE_DIR ..."
    
    tar --exclude="$SOURCE_DIR/.git" \
        --exclude="$SOURCE_DIR/.gitignore" \
        --exclude="$SOURCE_DIR/example" \
        --exclude="__pycache__" \
        --exclude="*.pyc" \
        --exclude="*.pyo" \
        --exclude="*.log" \
        --exclude="*.tmp" \
        -czf "$TAR_FILE" "$SOURCE_DIR" "$PWACT_DIR" "$PWDATA_DIR"
    
    green "步骤1: 快速压缩..."
    gzip -1 "$TAR_FILE"
    
    echo "✓ 生成: $GZ_FILE ($(du -h "$GZ_FILE" | cut -f1))"
}

# 第二步：Base64编码
step2_base64() {
    green "步骤2: Base64编码..."
    base64 "$GZ_FILE" > "$BASE64_FILE"
    echo "✓ 生成: $BASE64_FILE ($(du -h "$BASE64_FILE" | cut -f1))"
}

# 第三步：复制模板并追加base64内容
step3_create_script() {
    green "步骤3: 创建可执行脚本..."
    cp "$TEMPLATE_FILE" "$SH_FILE"
    cat "$BASE64_FILE" >> "$SH_FILE"
    chmod +x "$SH_FILE"
    echo "✓ 生成: $SH_FILE ($(du -h "$SH_FILE" | cut -f1))"
}

# 第四步：打包最终文件
step4_final_package() {
    green "步骤4: 创建最终压缩包..."
    
    # 检查check_offenv.sh是否存在，如果存在则包含它
    if [ -f "$CHECK_SCRIPT" ]; then
        tar -czvf "$FINAL_SH_TAR" "$SH_FILE" "$CHECK_SCRIPT"
    else
        tar -czvf "$FINAL_SH_TAR" "$SH_FILE"
        yellow "注意: 未包含 $CHECK_SCRIPT 文件"
    fi
    
    echo "✓ 生成: $FINAL_SH_TAR ($(du -h "$FINAL_SH_TAR" | cut -f1))"
}

# 清理临时文件
cleanup() {
    green "清理临时文件..."
    rm -f "$GZ_FILE" "$BASE64_FILE" "$SH_FILE"
    echo "✓ 临时文件已清理: $GZ_FILE, $BASE64_FILE, $SH_FILE"
}

# 显示总结信息
show_summary() {
    echo ""
    green "=========================================="
    green "           打包完成！"
    green "=========================================="
    echo "最终文件: $(pwd)/$FINAL_SH_TAR"
    echo "文件大小: $(du -h "$FINAL_SH_TAR" | cut -f1)"
    echo ""
    green "包含的文件:"
    echo "  - $SH_FILE (自解压脚本)"
    if [ -f "$CHECK_SCRIPT" ]; then
        echo "  - $CHECK_SCRIPT (环境检查脚本)"
    fi
    
    if [ "$CLEANUP_FLAG" = false ]; then
        yellow "临时文件保留:"
        echo "  - $GZ_FILE"
        echo "  - $BASE64_FILE"
        echo "  - $SH_FILE"
    fi
    green "=========================================="
}

# 主执行函数
main() {
    blue "开始打包流程..."
    echo "日期版本: $CURRENT_DATE"
    if [ "$CLEANUP_FLAG" = true ]; then
        yellow "清理模式: 启用 (打包完成后自动清理临时文件)"
    else
        yellow "清理模式: 禁用 (保留临时文件)"
    fi
    echo ""
    
    # 检查必要文件
    check_files
    
    # 执行各步骤
    step1_package
    step2_base64
    step3_create_script
    step4_final_package
    
    # 根据参数决定是否清理
    if [ "$CLEANUP_FLAG" = true ]; then
        cleanup
    fi
    
    show_summary
}

# 解析命令行参数
parse_arguments "$@"

# 执行主函数
main
