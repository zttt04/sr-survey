"""TinySR inference-speed profiler.

Measures, per input resolution (batch 1, fp16, H800):
  * end-to-end wall time (median / p95 over N iterations, after warmup)
  * stage split: VAE encode / transformer / VAE decode (CUDA event timed)
  * GPU busy time + kernel count + top kernels (torch kineto profiler)
  * peak allocated VRAM
Writes JSON to prof/tinysr.json.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/project/sponge/spongert/zt/TinySR")
os.chdir("/project/sponge/spongert/zt/TinySR")

import importlib.util
import torch
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig
from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict

ROOT = "/project/sponge/spongert/zt/TinySR/checkpoint/checkpoint"
PRETRAINED = f"{ROOT}/tinybackbone/prune-12-merge-tinysr"
VAE_PATH = f"{ROOT}/vae/separable"
LORA_DIR = f"{ROOT}/tinysr"
EMBED_DIR = "/project/sponge/spongert/zt/TinySR/dataset/default"
OUT = "/project/sponge/spongert/zt/prof"

# tile_sample straight from test/test_tinysr.py (keeps official behaviour, incl. tiling)
_spec = importlib.util.spec_from_file_location("tt", "/project/sponge/spongert/zt/TinySR/test/test_tinysr.py")
_tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tt)

WEIGHT_DTYPE = torch.float16


class Args:
    def __init__(self, device="cuda"):
        self.device = device
        self.latent_tiled_size = 64
        self.latent_tiled_overlap = 8
        self.rank = 64
        self.rank_vae = 64


def build():
    t0 = time.time()
    transformer = TinySD3Transformer2DModel.from_pretrained(
        PRETRAINED, subfolder="transformer", torch_dtype=WEIGHT_DTYPE,
        low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
    vae = AutoencoderTiny.from_pretrained(VAE_PATH, torch_dtype=WEIGHT_DTYPE)
    cfg = LoraConfig(r=64, lora_alpha=64, init_lora_weights="gaussian",
                     target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj",
                                     "linear", "linear_1", "linear_2", "net.2"])
    transformer.add_adapter(cfg)
    transformer.enable_adapters()
    sd = StableDiffusion3Pipeline.lora_state_dict(LORA_DIR, weight_name="transformer.safetensors")
    load_lora_state_dict(sd, transformer)
    vae = vae.to("cuda", dtype=WEIGHT_DTYPE).eval()
    transformer = transformer.to("cuda", dtype=WEIGHT_DTYPE).eval()
    load_s = time.time() - t0
    n_tf = sum(p.numel() for p in transformer.parameters())
    n_vae = sum(p.numel() for p in vae.parameters())
    return transformer, vae, load_s, n_tf, n_vae


def run_once(transformer, vae, timesteps, pooled, w, h, args):
    """One image: LR (h, w) -> SR (4h, 4w). Returns (t_encode, t_tf, t_decode) ms."""
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    dtype = WEIGHT_DTYPE
    x = torch.rand(1, 3, h, w, device="cuda", dtype=torch.float32)
    new_h, new_w = 4 * h, 4 * w
    new_h -= new_h % 8
    new_w -= new_w % 8
    with torch.no_grad():
        ev[4].record()
        pv = torch.nn.functional.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False)
        pv = (pv * 2 - 1).to("cuda", dtype=dtype)
        ev[0].record()
        latent = vae.encode(pv).latents * vae.config.scaling_factor
        latent = latent.to(dtype=dtype)
        ev[1].record()
        pred = _tt.tile_sample(latent, pv, transformer, timesteps, pooled, dtype, args)
        latent_stu = latent - pred
        ev[2].record()
        img = vae.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
    ev[3].record()
    torch.cuda.synchronize()
    return (
        ev[0].elapsed_time(ev[1]),
        ev[1].elapsed_time(ev[2]),
        ev[2].elapsed_time(ev[3]),
        ev[4].elapsed_time(ev[3]),
        tuple(img.shape),
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    res_list = [tuple(int(v) for v in r.lower().split("x")) for r in
                (sys.argv[1] if len(sys.argv) > 1 else "128x128,480x270,960x540").split(",")]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    transformer, vae, load_s, n_tf, n_vae = build()
    print(f"[load] {load_s:.1f}s  transformer {n_tf/1e6:.1f}M  vae {n_vae/1e6:.1f}M", flush=True)
    timesteps = torch.tensor([1000.], device="cuda", dtype=WEIGHT_DTYPE)
    pooled = torch.load(f"{EMBED_DIR}/pool_embeds.pt", map_location="cuda").to(dtype=WEIGHT_DTYPE)
    args = Args()
    _tt.args = args          # tile_sample's helper _gaussian_weights reads module-global `args`
    _tt.transformer = transformer  # ... and module-global `transformer`
    out = {"model": "TinySR", "gpu": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "dtype": "fp16", "batch": 1,
           "params_transformer_M": round(n_tf / 1e6, 1), "params_vae_M": round(n_vae / 1e6, 1),
           "load_s": round(load_s, 1), "resolutions": []}

    for (w, h) in [(r[0], r[1]) for r in res_list]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(2):  # warmup
                run_once(transformer, vae, timesteps, pooled, w, h, args)
            torch.cuda.synchronize()

            walls, enc, tf, dec, spans = [], [], [], [], []
            for _ in range(iters):
                t0 = time.perf_counter()
                e, t, d, span, shape = run_once(transformer, vae, timesteps, pooled, w, h, args)
                walls.append((time.perf_counter() - t0) * 1e3)
                enc.append(e); tf.append(t); dec.append(d); spans.append(span)

            # one traced iteration for kernel-level numbers
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA])
            t0 = time.perf_counter()
            with prof:
                run_once(transformer, vae, timesteps, pooled, w, h, args)
            traced_wall = (time.perf_counter() - t0) * 1e3
            evs = prof.key_averages()
            def _dev_us(ev):
                v = getattr(ev, "device_time_total", None)
                if v is None:
                    v = getattr(ev, "cuda_time_total", 0) or 0
                return v
            cuda_total = sum(_dev_us(e) for e in evs) / 1e3  # ms
            top = sorted(evs, key=_dev_us, reverse=True)[:8]
            peak = torch.cuda.max_memory_allocated() / 2**20

            rec = {
                "lr": f"{w}x{h}", "sr": f"{shape[3]}x{shape[2]}",
                "wall_median_ms": round(statistics.median(walls), 1),
                "wall_min_ms": round(min(walls), 1),
                "wall_p95_ms": round(sorted(walls)[max(0, int(0.95 * len(walls)) - 1)], 1),
                "wall_all_ms": [round(v, 1) for v in walls],
                "stage_ms_median": {
                    "vae_encode": round(statistics.median(enc), 1),
                    "transformer": round(statistics.median(tf), 1),
                    "vae_decode": round(statistics.median(dec), 1),
                },
                "gpu_span_ms": round(statistics.median(spans), 1),
                "gpu_busy_ms": round(cuda_total, 1),
                "gpu_util_pct": round(100 * cuda_total / traced_wall, 1),
                "kernel_count": len(evs),
                "traced_wall_ms": round(traced_wall, 1),
                "peak_vram_MiB": round(peak, 0),
                "top_kernels": [
                    {"name": e.key[:70], "cuda_ms": round(_dev_us(e) / 1e3, 2),
                     "pct": round(100 * _dev_us(e) / 1e3 / cuda_total, 1),
                     "calls": e.count} for e in top if _dev_us(e) > 0
                ],
            }
            print(f"[{rec['lr']} -> {rec['sr']}] wall {rec['wall_median_ms']}ms "
                  f"(enc {rec['stage_ms_median']['vae_encode']} / tf {rec['stage_ms_median']['transformer']}"
                  f" / dec {rec['stage_ms_median']['vae_decode']}) gpu {rec['gpu_busy_ms']}ms "
                  f"vram {rec['peak_vram_MiB']}MiB", flush=True)
            out["resolutions"].append(rec)
        except RuntimeError as exc:
            print(f"[{w}x{h}] FAILED: {str(exc)[:200]}", flush=True)
            out["resolutions"].append({"lr": f"{w}x{h}", "error": str(exc)[:300]})
            torch.cuda.empty_cache()

    with open(f"{OUT}/tinysr.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[done] ->", f"{OUT}/tinysr.json", flush=True)


if __name__ == "__main__":
    main()
