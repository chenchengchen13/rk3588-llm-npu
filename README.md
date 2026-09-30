# RK3588 NPU LLM 部署实践

在瑞芯微 RK3588（6 TOPS NPU）上从 HuggingFace 权重出发，完成 LLM 量化转换、上板部署与性能基准测试的完整记录。包含 DeepSeek-R1-Distill-Qwen-1.5B 与 Qwen2.5-0.5B-Instruct 两个模型，以及一份 rkllm-runtime 版本兼容 bug 的定位过程。

## 硬件 / 软件环境

| 项 | 配置 |
|---|---|
| 开发板 | LubanCat-4（RK3588S，16GB LPDDR4x） |
| NPU | 3 核，锁频 1 GHz，rknpu 驱动 0.9.8 |
| 转换端 | WSL2 + rkllm-toolkit 1.2.2 / 1.2.3 |
| 板端 runtime | librkllmrt 1.2.2 与 1.2.3 并存（原因见踩坑记录） |
| CPU 基线 | llama.cpp（ggml-org master，板端自编译） |

## 实测数据

### DeepSeek-R1-Distill-Qwen-1.5B（W8A8，3 NPU 核）

| 路径 | prefill | decode | 内存 | 备注 |
|---|---|---|---|---|
| NPU（自转模型，runtime 1.2.2） | 96–144 tok/s | **9.9–10.3 tok/s** | 1773 MB | 思维链 `<think>` 输出完整 |
| CPU（llama.cpp Q4_K_M，4 线程） | 56.2 tok/s（pp128） | **22.1 tok/s**（tg32） | — | — |

### Qwen2.5-0.5B-Instruct（W8A8，3 NPU 核）

| 路径 | prefill | decode | 内存 |
|---|---|---|---|
| NPU（runtime 1.2.3） | **420–459 tok/s** | 21.7–23.0 tok/s | 695 MB |
| CPU（llama.cpp Q4_0，4 线程） | 358.6 tok/s（pp128） | **60.2 tok/s**（tg32） | — |

### 核心结论

**端侧 LLM 推理的瓶颈在内存带宽而非算力。** NPU 只在 prefill（计算密集）有优势（+20%～2.6×），decode（访存密集）反而比 CPU 慢 50% 以上——6 TOPS 算力对 decode 无加成，且 W8A8（8 bit）比 Q4_0/Q4_K_M（4 bit）的权重读取量约大一倍。另外实测 8 线程 decode 比 4 线程更慢：A76+A55 大小核在访密密集负载下的调度损耗。

## 踩坑记录（排障过程）

### 1. librkllmrt 1.3.1 + 驱动 0.9.8 → SIGBUS
升级 runtime 到 1.3.1 后模型加载直接总线错误。回退 1.2.3 恢复。教训：runtime 版本必须与转换所用 toolkit 版本及内核驱动匹配。

### 2. DeepSeek-R1-Distill 输出崩坏成 `[PAD151935]` 重复（本仓库重点）
**现象**：模型开头几个 token 正常，随后崩坏为 `[PAD151935]` 无限重复。

**排查路径**：
1. 怀疑量化校准数据 → 换三种校准集（官方默认 19 条英文 / 自建中文 140 条 / 官方 generate_data_quant.py 自生成）重转，全部同样崩坏；
2. 验证 HF 源模型完整性 → 正常；
3. **决定性实验**：radxa 官方预转换模型（toolkit 1.2.2 转换）在 runtime 1.2.3 下同样崩坏 → 排除转换侧问题；
4. 换用 radxa 仓库附带的 **runtime 1.2.2** → 输出完全正常。

**结论**：rkllm-runtime **1.2.3** 对 DeepSeek-R1-Distill-Qwen-1.5B（Qwen2 架构蒸馏版，`tie_word_embeddings=False`、`sliding_window`、双 BOS id）存在兼容性 bug，与 toolkit 版本和校准数据无关。社区同症状 issue：airockchip/rknn-llm#467。

**解法**：板端双 runtime 并存——DS 模型用 `LD_PRELOAD=./librkllmrt.so.1.2.2` 运行，Qwen 模型继续用系统 1.2.3；转换侧对应降级 toolkit 1.2.2。

### 3. WSL2 静默杀死后台转换进程
两次过夜转换无声失败，日志无报错。根因：WSL2 默认 `vmIdleTimeout=60s`，无交互会话时虚拟机自动关闭。修复：`.wslconfig` 写入 `[wsl2] vmIdleTimeout=-1`。

### 4. WSL 内 torch 缺 iJIT 符号
intel-openmp 2025 移除了 iJIT 相关符号，torch import 即崩。解法：`LD_PRELOAD` 一个提供该符号的空 stub。

## 文件说明

| 文件 | 说明 |
|---|---|
| `bench/rkllm_bench.cpp` | 板端基准工具：模型加载计时 + prefill/decode tok/s + 内存统计，支持多轮独立测试（自动清 KV cache） |
| `convert/convert_one.py` | 通用转换脚本（Qwen2.5-0.5B 等），W8A8 / normal / rk3588 / 3 核 |
| `convert/convert_ds_122.py` | DeepSeek-R1-Distill-1.5B 转换脚本（toolkit 1.2.2 + 官方校准配方） |
| `convert/gen_calib_zh.py` | 中文校准数据集生成器（140 条：数学推理 / 常识 QA / 英文 CoT / 指令跟随） |

## 复现要点

```bash
# 板端 benchmark（DS 模型需 LD_PRELOAD 指定 runtime 1.2.2）
LD_PRELOAD=./librkllmrt.so.1.2.2 ./rkllm_bench model.rkllm 512 4096 "你的 prompt" 3

# 输出示例：
# [LOAD] rkllm_init 2200.3 ms
# [PERF run 0] prefill 21 tokens 150.4 ms (139.62 tok/s) | decode 127 tokens 12364.5 ms (10.27 tok/s) | mem 1771.5 MB
```

模型文件（.rkllm，约 2 GB）不上传仓库。radxa 预转换版本见 ModelScope `radxa/DeepSeek-R1-Distill-Qwen-1.5B_RKLLM`；自转按 `convert/` 下脚本执行即可。

## 许可

代码 MIT。模型权重遵循各自原始许可（DeepSeek-R1 蒸馏模型：MIT；Qwen2.5：Apache 2.0）。
