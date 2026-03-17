# MatPL 代码结构思维导图

下面提供一个面向新人的「从用户交互到内部模块」的思维导图。

## 1) 整体思维导图（Mermaid）

```mermaid
mindmap
  root((MatPL / PWMLFF))
    用户交互层
      CLI 入口
        main.py
        MatPL -h / train / test / infer / model_devi
      输入
        JSON 配置文件
        结构文件(pwmat/config extxyz lammps/dump)
        checkpoint 或 nep.txt
      输出
        训练模型.ckpt
        推理结果(txt/csv)
        导出力场(txt/libtorch)

    顶层调度层(main.py)
      命令分发
        train
        test
        extract_ff
        compress
        script
        totxt
        infer
        model_devi
      模型类型分发
        DP
        NN
        LINEAR
        NEP

    参数与工作目录层(src/user)
      InputParam
        解析 model_type/atom_type
        解析 descriptor/model/optimizer
        构建 WorkFileStructure
      *_work.py
        dp_work.py
        nn_work.py
        linear_work.py
        nep_work.py
      辅助工具
        ckpt_extract.py
        ckpt_compress.py
        infer_main.py

    训练与推理核心层(src/PWMLFF + src/model + src/optimizer)
      网络流程
        dp_network.py
        nn_network.py
        nep_network.py
        linear_regressor.py
      模型定义
        src/model/*
      优化器
        ADAM/ADAMW
        LKF/GKF
      训练器实现
        *_mods/*_trainer.py

    数据与特征层(src/pre_data + src/feature)
      数据读取与整理
        dpuni_data_loader.py
        数据格式适配(pwmat/movement pwmlff/npy)
      邻居/特征计算
        NEP GPU/CPU 特征模块
        chebyshev 特征模块

    高性能算子层(src/op)
      PyTorch C++/CUDA 扩展
        calculateForce
        calculateVirial
        calculateCompress
      编译入口
        src/op/setup.py

    测试与示例层
      自动化测试
        src/test/auto_test.py
        src/test/README.md
      示例配置
        example/Cu
        example/LiSi
        example/Ag-Au-D3
```

## 2) 用户交互到调用链（简图）

```mermaid
flowchart TD
  U[用户: MatPL train input.json] --> M[main.py]
  M --> P[InputParam 解析 JSON]
  P --> W[dp_work/nn_work/nep_work]
  W --> N[dp_network/nn_network/nep_network]
  N --> D[UniDataset / pre_data 数据加载]
  N --> O[optimizer]
  N --> K[src/op CUDA算子]
  N --> C[checkpoint 与日志输出]
```

## 3) 模块关系速记

- `main.py` 是总入口：命令 + 模型类型双分发。
- `src/user` 是编排层：参数抽象、任务流程、模型导出/压缩/推理工具。
- `src/PWMLFF` 是训练推理主线：把数据、模型、优化器串起来。
- `src/model` 和 `src/optimizer` 分别承载模型结构与优化算法。
- `src/pre_data` 和 `src/feature` 负责数据预处理、邻域搜索与特征生成。
- `src/op` 提供 CUDA 扩展算子，是性能关键路径。
- `src/test` + `example` 用于回归验证和新手跑通样例。

## 4) 建议阅读顺序

1. `main.py`（先看入口和命令分发）
2. `src/user/input_param.py`（理解 JSON 字段如何映射到代码对象）
3. `src/user/dp_work.py`（看一条完整 train/test 业务流程）
4. `src/PWMLFF/dp_network.py`（看训练器核心）
5. `src/op/setup.py`（理解高性能算子如何接入）
6. `src/test/auto_test.py` + `example/*`（跑通并做回归）
