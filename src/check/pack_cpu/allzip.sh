#!/bin/bash

rm -rf *part_a* matpl_cpu-2025.3.sh.tar.gz MatPL_cpu-2025.3.tar.gz.base64 matpl_cpu-2025.3.sh MatPL_cpu-2025.3.tar.gz

# 打包环境
# conda pack -n matpl_cpu-2025.3

# 将打包好的环境和 MatPL 目录打包成 tar.gz 文件
tar -czf MatPL_cpu-2025.3.tar.gz matpl_cpu-2025.3.tar.gz MatPL_cpu-2025.3 lammps-stable

# 将 tar.gz 文件编码成 base64
base64 MatPL_cpu-2025.3.tar.gz > MatPL_cpu-2025.3.tar.gz.base64

# 复制模板脚本并添加 base64 编码的 tar.gz 数据
cp matpl_cpu-2025.3.sh.template matpl_cpu-2025.3.sh
cat MatPL_cpu-2025.3.tar.gz.base64 >> matpl_cpu-2025.3.sh

# 打包最终的脚本
tar -czvf matpl_cpu-2025.3.sh.tar.gz matpl_cpu-2025.3.sh check_offenv_cpu.sh

# 分割
split -b 800M matpl_cpu-2025.3.sh.tar.gz matpl_cpu-2025.3.sh.tar.gz.part_

# copy to /share
cp matpl_cpu-2025.3.sh.tar.gz.part_* /share/public/PWMLFF_test_data/matpl-pack/2025.3-cpu

