"""DLSS-NR: RealSR benchmark outputs + H800 speed profile.

Part 1 (metrics): 100 RealSR LR crops -> bilinear 4x upscale -> DLSS-NR (1x) -> 512x512 output.
Part 2 (speed):   steady-state per-resolution latency (CUDA events), peak VRAM, kineto top kernels.

Runs with DLSSNR_COMPILE=0: the torch.compile path mis-compiles on CUDA (output drifts to 21.5 dB
against NVIDIA's reference, while the eager path reproduces the CPU result to the bit, 33.78 dB).
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
import time

os.environ["DLSSNR_COMPILE"] = "0"
REPO = "/project/sponge/spongert/zt/dlssnr-pytorch"
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import torch
from PIL import Image

from dlssnr.io import Controls, tensor_to_frame
from dlssnr.model import NRNet
from dlssnr.weights import Weights

ROOT = "/project/sponge/spongert/zt"
HR = f"{ROOT}/testsets/StableSR_testsets/RealSRVal_crop128/test_HR"
LR = f"{ROOT}/testsets/StableSR_testsets/RealSRVal_crop128/test_LR"
OUT = f"{ROOT}/eval_out/realsr/dlssnr"
PROF = f"{ROOT}/prof"


def timing(net, frame, iters=5, warmup=2):
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    with torch.no_grad():
        for _ in range(warmup):
            net(frame, Controls())
        torch.cuda.synchronize()
        for i in range(iters):
            ev[i][0].record()
            net(frame, Controls())
            ev[i][1].record()
        torch.cuda.synchronize()
    return [a.elapsed_time(b) for a, b in ev]


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PROF, exist_ok=True)
    device = torch.device("cuda")
    weights = Weights.load(f"{REPO}/weights.safetensors")
    out = {"model": "DLSS-NR", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "dtype": "fp16 storage / fp32 accumulate (reference arithmetic, DLSSNR_COMPILE=0)",
           "batch": 1, "note": "1x network: output resolution == input resolution"}

    # ---------- part 1: RealSR outputs + per-image timing ----------
    names = sorted(glob.glob(f"{LR}/*.png"))
    net = NRNet(weights, 512, 512).to(device)
    per_image = []
    with torch.no_grad():
        # warm up on the first frame (twice), then run and save all 100
        first = np.asarray(Image.open(names[0]).convert("RGB").resize((512, 512), Image.BILINEAR), dtype=np.uint8)
        net(first, Controls()); net(first, Controls())
        torch.cuda.synchronize()
        for i, p in enumerate(names):
            stem = os.path.splitext(os.path.basename(p))[0]
            lr = Image.open(p).convert("RGB").resize((512, 512), Image.BILINEAR)
            frame = np.asarray(lr, dtype=np.uint8)
            torch.cuda.synchronize()
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            result = net(frame, Controls())
            e1.record()
            torch.cuda.synchronize()
            per_image.append(e0.elapsed_time(e1))
            Image.fromarray(result).save(f"{OUT}/{stem}.png")
    torch.cuda.empty_cache()
    out["realsr"] = {
        "images": len(per_image),
        "median_ms": round(statistics.median(per_image), 1),
        "min_ms": round(min(per_image), 1),
        "max_ms": round(max(per_image), 1),
        "resolution": "512x512 (bilinear 4x upscale of the 128x128 LR)",
    }
    print(f"[RealSR] {len(per_image)} images @512x512: median {out['realsr']['median_ms']} ms "
          f"(min {out['realsr']['min_ms']}, max {out['realsr']['max_ms']})", flush=True)
    del net

    # ---------- part 2: speed at several resolutions ----------
    out["resolutions"] = []
    rng = np.random.default_rng(0)
    for (w, h) in [(512, 512), (1920, 1080), (2560, 1440), (3840, 2160)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        net = NRNet(weights, w, h).to(device)
        frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        try:
            times = timing(net, frame, iters=5, warmup=2)
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
            with prof:
                net(frame, Controls())
                torch.cuda.synchronize()

            def _dev_us(ev):
                v = getattr(ev, "device_time_total", None)
                return v if v is not None else (getattr(ev, "cuda_time_total", 0) or 0)

            evs = prof.key_averages()
            total = sum(_dev_us(e) for e in evs) / 1e3
            top = sorted(evs, key=_dev_us, reverse=True)[:8]
            peak = torch.cuda.max_memory_allocated() / 2**30
            rec = {
                "resolution": f"{w}x{h}",
                "median_ms": round(statistics.median(times), 1),
                "min_ms": round(min(times), 1),
                "all_ms": [round(t, 1) for t in times],
                "peak_vram_GiB": round(peak, 2),
                "kernel_count": len(evs),
                "gpu_ms_sum": round(total, 1),
                "top_kernels": [
                    {"name": e.key[:70], "cuda_ms": round(_dev_us(e) / 1e3, 2),
                     "pct": round(100 * _dev_us(e) / 1e3 / total, 1), "calls": e.count}
                    for e in top if _dev_us(e) > 0 and not e.key.startswith("aten::")
                ],
            }
            print(f"[{rec['resolution']}] median {rec['median_ms']} ms  peak {rec['peak_vram_GiB']} GiB",
                  flush=True)
            out["resolutions"].append(rec)
        except RuntimeError as exc:
            print(f"[{w}x{h}] FAILED: {str(exc)[:150]}", flush=True)
            out["resolutions"].append({"resolution": f"{w}x{h}", "error": str(exc)[:200]})
        del net

    with open(f"{PROF}/dlssnr.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[done] ->", f"{PROF}/dlssnr.json", flush=True)


if __name__ == "__main__":
    main()
