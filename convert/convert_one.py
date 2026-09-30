"""RKLLM 转换脚本：python convert_one.py <hf模型目录> <输出rkllm路径>
目标平台 rk3588，W8A8 量化，normal 算法，3 NPU 核。"""
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from rkllm.api import RKLLM

modelpath = sys.argv[1]
outpath = sys.argv[2]
dtype = sys.argv[3] if len(sys.argv) > 3 else "float32"

DATASET = os.path.expanduser("~/rknn-llm/examples/rkllm_api_demo/export/data_quant.json")

llm = RKLLM()
t0 = time.time()
print(f"[load] {modelpath} dtype={dtype}", flush=True)
ret = llm.load_huggingface(model=modelpath, model_lora=None, device="cpu",
                           dtype=dtype, custom_config=None, load_weight=True)
if ret != 0:
    print("Load model failed!")
    sys.exit(ret)
print(f"[load done] {time.time()-t0:.0f}s", flush=True)

t1 = time.time()
ret = llm.build(do_quantization=True, optimization_level=1,
                quantized_dtype="W8A8", quantized_algorithm="normal",
                target_platform="rk3588", num_npu_core=3,
                extra_qparams=None, dataset=DATASET,
                hybrid_rate=0, max_context=4096)
if ret != 0:
    print("Build model failed!")
    sys.exit(ret)
print(f"[build done] {time.time()-t1:.0f}s", flush=True)

ret = llm.export_rkllm(outpath)
if ret != 0:
    print("Export model failed!")
    sys.exit(ret)
print(f"[export done] {outpath} total {time.time()-t0:.0f}s", flush=True)
