"""DeepSeek-R1-Distill-Qwen-1.5B 转换：toolkit 1.2.2 + 官方自生成校准数据。
注意：该模型必须用 toolkit/runtime 1.2.2 组合，runtime 1.2.3 存在兼容 bug（详见 README）。
配套板端运行方式：LD_PRELOAD=./librkllmrt.so.1.2.2 ./rkllm_bench ..."""
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from rkllm.api import RKLLM

MODEL = os.path.expanduser("~/models/DeepSeek-R1-Distill-Qwen-1.5B")
OUT = os.path.expanduser("~/models/DS_W8A8_tk122_RK3588.rkllm")
DATASET = os.path.expanduser("~/rkllm_dl/data_quant_official.json")

llm = RKLLM()
t0 = time.time()
print(f"[load] {MODEL}", flush=True)
ret = llm.load_huggingface(model=MODEL, model_lora=None, device="cpu",
                           dtype="float32", custom_config=None, load_weight=True)
if ret != 0:
    print("Load model failed!")
    sys.exit(ret)
print(f"[load done] {time.time()-t0:.0f}s", flush=True)

t1 = time.time()
ret = llm.build(do_quantization=True, optimization_level=1,
                quantized_dtype="w8a8", quantized_algorithm="normal",
                target_platform="rk3588", num_npu_core=3,
                extra_qparams=None, dataset=DATASET,
                hybrid_rate=0, max_context=4096)
if ret != 0:
    print("Build model failed!")
    sys.exit(ret)
print(f"[build done] {time.time()-t1:.0f}s", flush=True)

ret = llm.export_rkllm(OUT)
if ret != 0:
    print("Export model failed!")
    sys.exit(ret)
print(f"[export done] {OUT} total {time.time()-t0:.0f}s", flush=True)
