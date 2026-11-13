# 推理速度实测脚本

测量条件：单卡 H800 80 GB，fp16，batch = 1，预热 2 次后取 5 次中位数；
计时只包含模型计算（不含读图 / 存图 / 色彩对齐）。

| 文件 | 用途 |
|---|---|
| `prof_tinysr.py` | TinySR：分段（VAE 编码 / transformer / VAE 解码）+ 峰值显存 + kineto kernel 统计 |
| `prof_sinsr.py` | SinSR：官方 `Sampler.inference` 路径（256 / 步长 224 切块），记录块数与单块耗时 |

两者都用 `torch.profiler`（kineto，与 Nsight Systems 同一套 CUPTI 后端）抓 kernel 级数据。

```bash
# 依赖：torch 2.6 (cu124) + diffusers 0.29.1 + peft 0.15.0 + transformers 4.49.0
python prof_tinysr.py 128x128,480x270,960x540 5     # -> prof/tinysr.json
python prof_sinsr.py  128x128,480x270,960x540 3     # -> prof/sinsr.json
```

MiniMax H3-Image 的数字取自服务端日志（`Pixel data generated successfully in ... seconds`），
DLSS-NR 是 CPU 上的 PyTorch 复现，2 者都无法用同一套脚本测。

说明：容器里既没有 `nsys` 也没有 GPU 监控权限（非 root，CUDA 源里的 nsight-systems 包缺 target 目录装不上），
所以用 PyTorch 自带的 kineto profiler 取 kernel 级数据，GPU 总时间用 CUDA event 交叉验证。
