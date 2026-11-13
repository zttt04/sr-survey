"""SinSR (one-step) inference-speed profiler.

Per LR resolution (batch 1, fp16, H800):
  * end-to-end wall time through the official Sampler.inference path (incl. 256 chop tiling)
  * per-tile sample_func time (encode + 1 model eval + decode) and tile count
  * GPU busy time / kernel count / top kernels (torch kineto), peak VRAM
Writes prof/sinsr.json.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time

SINSR = "/project/sponge/spongert/zt/SinSR"
sys.path.insert(0, SINSR)
os.chdir(SINSR)

import types

import torchvision.transforms.functional as _tvf

_compat = types.ModuleType("torchvision.transforms.functional_tensor")
_compat.rgb_to_grayscale = _tvf.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _compat
_eval = types.ModuleType("evaluate")
_eval.load = lambda *a, **k: None
_eval.evaluate = lambda *a, **k: None
sys.modules.setdefault("evaluate", _eval)

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from sampler import Sampler

OUT = "/project/sponge/spongert/zt/prof"


def build():
    t0 = time.time()
    configs = OmegaConf.load("./configs/SinSR.yaml")
    configs.model.ckpt_path = "./weights/SinSR_v1.pth"
    configs.diffusion.params.timestep_respacing = None   # same as inference.py with no --infer_steps
    configs.diffusion.params.sf = 4
    configs.autoencoder.ckpt_path = "./weights/autoencoder_vq_f4.pth"
    sampler = Sampler(configs, chop_size=256, chop_stride=224, chop_bs=1,
                      use_fp16=True, seed=42, ddim=False)
    load_s = time.time() - t0
    n_model = sum(p.numel() for p in sampler.model.parameters())
    n_ae = sum(p.numel() for p in sampler.autoencoder.parameters()) if sampler.autoencoder is not None else 0
    return sampler, load_s, n_model, n_ae


def make_lr(path, w, h, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    res_list = [tuple(int(v) for v in r.lower().split("x")) for r in
                (sys.argv[1] if len(sys.argv) > 1 else "128x128,480x270,960x540").split(",")]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    sampler, load_s, n_model, n_ae = build()
    print(f"[load] {load_s:.1f}s  model {n_model/1e6:.1f}M  autoencoder {n_ae/1e6:.1f}M", flush=True)

    # per-tile timing hook
    calls = []
    orig_sample = sampler.sample_func

    def timed_sample(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = orig_sample(*a, **k)
        torch.cuda.synchronize()
        calls.append((time.perf_counter() - t) * 1e3)
        return r

    sampler.sample_func = timed_sample

    out = {"model": "SinSR", "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "dtype": "fp16", "batch": 1,
           "params_model_M": round(n_model / 1e6, 1), "params_autoencoder_M": round(n_ae / 1e6, 1),
           "load_s": round(load_s, 1), "chop_size": 256, "chop_stride": 224, "resolutions": []}

    for (w, h) in res_list:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tmp = tempfile.mkdtemp()
        src = f"{tmp}/in"; dst = f"{tmp}/out"
        os.makedirs(src, exist_ok=True)
        p = f"{src}/probe_{w}x{h}.png"
        make_lr(p, w, h)
        try:
            for _ in range(1):  # warmup
                calls.clear()
                sampler.inference(src, f"{tmp}/warm", bs=1, noise_repeat=False, one_step=True)
                torch.cuda.synchronize()

            walls, tiles = [], []
            for _ in range(iters):
                calls.clear()
                t0 = time.perf_counter()
                sampler.inference(src, dst, bs=1, noise_repeat=False, one_step=True)
                torch.cuda.synchronize()
                walls.append((time.perf_counter() - t0) * 1e3)
                tiles.append(list(calls))

            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA])
            t0 = time.perf_counter()
            with prof:
                sampler.inference(src, dst, bs=1, noise_repeat=False, one_step=True)
                torch.cuda.synchronize()
            traced_wall = (time.perf_counter() - t0) * 1e3
            evs = prof.key_averages()
            def _dev_us(ev):
                v = getattr(ev, "device_time_total", None)
                if v is None:
                    v = getattr(ev, "cuda_time_total", 0) or 0
                return v
            cuda_total = sum(_dev_us(e) for e in evs) / 1e3
            top = sorted(evs, key=_dev_us, reverse=True)[:8]
            peak = torch.cuda.max_memory_allocated() / 2**20

            sr_shape = Image.open(f"{dst}/probe_{w}x{h}.png").size
            rec = {
                "lr": f"{w}x{h}", "sr": f"{sr_shape[0]}x{sr_shape[1]}",
                "wall_median_ms": round(statistics.median(walls), 1),
                "wall_min_ms": round(min(walls), 1),
                "wall_all_ms": [round(v, 1) for v in walls],
                "tiles": len(tiles[0]),
                "tile_ms_median": round(statistics.median([t for run in tiles for t in run]), 1) if tiles[0] else None,
                "gpu_busy_ms": round(cuda_total, 1),
                "gpu_util_pct": round(100 * cuda_total / traced_wall, 1),
                "kernel_count": len(evs),
                "traced_wall_ms": round(traced_wall, 1),
                "peak_vram_MiB": round(peak, 0),
                "top_kernels": [
                    {"name": e.key[:70], "cuda_ms": round(_dev_us(e) / 1e3, 2),
                     "pct": round(100 * _dev_us(e) / 1e3 / cuda_total, 1), "calls": e.count}
                    for e in top if _dev_us(e) > 0
                ],
            }
            print(f"[{rec['lr']} -> {rec['sr']}] wall {rec['wall_median_ms']}ms "
                  f"tiles {rec['tiles']} x {rec['tile_ms_median']}ms  gpu {rec['gpu_busy_ms']}ms "
                  f"vram {rec['peak_vram_MiB']}MiB", flush=True)
            out["resolutions"].append(rec)
        except Exception as exc:  # noqa: BLE001
            print(f"[{w}x{h}] FAILED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            out["resolutions"].append({"lr": f"{w}x{h}", "error": f"{type(exc).__name__}: {exc}"[:300]})
            torch.cuda.empty_cache()

    with open(f"{OUT}/sinsr.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[done] ->", f"{OUT}/sinsr.json", flush=True)


if __name__ == "__main__":
    main()
