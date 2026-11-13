# 推理速度实测脚本

测量条件：单卡 H800 80 GB，fp16，batch = 1，预热 2 次后取 5 次中位数；
计时只包含模型计算（不含读图 / 存图 / 色彩对齐）。

| 文件 | 用途 |
|---|---|
| `prof_tinysr.py` | TinySR：分段（VAE 编码 / transformer / VAE 解码）+ 峰值显存 + kineto kernel 统计 |
| `prof_sinsr.py` | SinSR：官方 `Sampler.inference` 路径（256 / 步长 224 切块），记录块数与单块耗时 |
| `prof_dlssnr.py` | DLSS-NR：跑 RealSR 100 张（双线性 4× 放大 → 1× 网络）并出图，同时测 4 档分辨率的延迟与峰值显存 |

两者都用 `torch.profiler`（kineto，与 Nsight Systems 同一套 CUPTI 后端）抓 kernel 级数据。

```bash
# 依赖：torch 2.6 (cu124) + diffusers 0.29.1 + peft 0.15.0 + transformers 4.49.0
python prof_tinysr.py 128x128,480x270,960x540 5     # -> prof/tinysr.json
python prof_sinsr.py  128x128,480x270,960x540 3     # -> prof/sinsr.json
python prof_dlssnr.py                              # -> prof/dlssnr.json + eval_out/realsr/dlssnr/*.png
```

MiniMax H3-Image 的数字取自服务端日志（`Pixel data generated successfully in ... seconds`），它不是一个能本地跑的 pipeline，所以没走这套脚本。

DLSS-NR 必须设置 `DLSSNR_COMPILE=0`：仓库默认开启的 `torch.compile` 在 CUDA 上会编错，
输出对 NVIDIA 参考只有 21.5 dB（eager 路径 33.78 dB，与 CPU 结果逐位一致）。

说明：容器里既没有 `nsys` 也没有 GPU 监控权限（非 root，CUDA 源里的 nsight-systems 包缺 target 目录装不上），
所以用 PyTorch 自带的 kineto profiler 取 kernel 级数据，GPU 总时间用 CUDA event 交叉验证。
