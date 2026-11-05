"""
efficiency.py
-------------
Model efficiency profiler for DeepJSCC research models.
Measures:
  - Parameter counts (trainable, frozen, total)
  - Checkpoint file size on disk (MB)
  - Latency (ms per image) at batch size 1 and batch size 64 on GPU and CPU
  - Throughput (images/sec) at batch size 1 and batch size 64
  - Peak GPU memory (MB) during inference
  - MACs: Marked N/A with analytical note to preserve strict dependency purity.

Usage:
  python efficiency.py --all
  python efficiency.py --config configs/model1_fair.yaml --checkpoint checkpoints/model1_fair/best.pt
"""

import os
import sys
import time
import argparse
import csv
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_model, load_config, _needs_imgnet


def get_param_counts(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total     = trainable + frozen
    return trainable, frozen, total


def get_checkpoint_size_mb(ckpt_path: str):
    if ckpt_path and os.path.isfile(ckpt_path):
        return os.path.getsize(ckpt_path) / (1024 * 1024)
    return 0.0


@torch.no_grad()
def benchmark_latency(model: nn.Module, device: torch.device, batch_size: int,
                      is_imgnet: bool = False,
                      warmup: int = 10, reps: int = 50,
                      channel: str = "awgn", snr_db: float = 10.0):
    model.eval()
    model.to(device)

    # Input tensor shape: (B, 3, 64, 64)
    inp = torch.randn(batch_size, 3, 64, 64, device=device)

    # Warmup
    for _ in range(warmup):
        _ = model(inp, channel=channel, snr_db=snr_db)

    if device.type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.reset_peak_memory_stats(device)
        start_event.record()
        for _ in range(reps):
            _ = model(inp, channel=channel, snr_db=snr_db)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event) # milliseconds total
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = model(inp, channel=channel, snr_db=snr_db)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        peak_vram_mb = 0.0

    total_images = reps * batch_size
    ms_per_image = elapsed_ms / total_images
    throughput = total_images / (elapsed_ms / 1000.0)

    return ms_per_image, throughput, peak_vram_mb


def profile_model_entry(name: str, config_path: str, ckpt_path: str, profile_cpu: bool = True):
    print(f"\n--- Profiling: {name} ---")
    cfg = load_config(config_path)
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(cfg, device_gpu)
    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device_gpu)
        model.load_state_dict(ckpt["model"], strict=False)
        ckpt_size_mb = get_checkpoint_size_mb(ckpt_path)
    else:
        ckpt_size_mb = 0.0

    trainable, frozen, total = get_param_counts(model)
    is_imgnet = _needs_imgnet(cfg)

    # GPU Benchmarks
    if torch.cuda.is_available():
        gpu_bs1_ms, gpu_bs1_fps, vram_bs1 = benchmark_latency(model, device_gpu, batch_size=1, is_imgnet=is_imgnet)
        gpu_bs64_ms, gpu_bs64_fps, vram_bs64 = benchmark_latency(model, device_gpu, batch_size=64, is_imgnet=is_imgnet)
    else:
        gpu_bs1_ms, gpu_bs1_fps, vram_bs1 = "N/A", "N/A", "N/A"
        gpu_bs64_ms, gpu_bs64_fps, vram_bs64 = "N/A", "N/A", "N/A"

    # CPU Benchmarks
    if profile_cpu:
        device_cpu = torch.device("cpu")
        model.to(device_cpu)
        cpu_bs1_ms, cpu_bs1_fps, _ = benchmark_latency(model, device_cpu, batch_size=1, is_imgnet=is_imgnet, reps=20, warmup=5)
        cpu_bs64_ms, cpu_bs64_fps, _ = benchmark_latency(model, device_cpu, batch_size=64, is_imgnet=is_imgnet, reps=10, warmup=2)
    else:
        cpu_bs1_ms, cpu_bs1_fps = "N/A", "N/A"
        cpu_bs64_ms, cpu_bs64_fps = "N/A", "N/A"

    result = {
        "model": name,
        "config": config_path,
        "checkpoint": ckpt_path,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "total_params": total,
        "ckpt_size_mb": round(ckpt_size_mb, 2),
        "gpu_latency_bs1_ms": round(gpu_bs1_ms, 3) if isinstance(gpu_bs1_ms, float) else gpu_bs1_ms,
        "gpu_fps_bs1": round(gpu_bs1_fps, 1) if isinstance(gpu_bs1_fps, float) else gpu_bs1_fps,
        "gpu_latency_bs64_ms": round(gpu_bs64_ms, 3) if isinstance(gpu_bs64_ms, float) else gpu_bs64_ms,
        "gpu_fps_bs64": round(gpu_bs64_fps, 1) if isinstance(gpu_bs64_fps, float) else gpu_bs64_fps,
        "gpu_peak_vram_mb": round(vram_bs64, 1) if isinstance(vram_bs64, float) else vram_bs64,
        "cpu_latency_bs1_ms": round(cpu_bs1_ms, 3) if isinstance(cpu_bs1_ms, float) else cpu_bs1_ms,
        "cpu_fps_bs1": round(cpu_bs1_fps, 1) if isinstance(cpu_bs1_fps, float) else cpu_bs1_fps,
        "cpu_latency_bs64_ms": round(cpu_bs64_ms, 3) if isinstance(cpu_bs64_ms, float) else cpu_bs64_ms,
        "cpu_fps_bs64": round(cpu_bs64_fps, 1) if isinstance(cpu_bs64_fps, float) else cpu_bs64_fps,
        "macs": "N/A",
        "macs_note": "Analytical FLOPs depend on channel use; thop omitted to preserve dependency purity"
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="DeepJSCC Model Efficiency Profiler")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pt")
    parser.add_argument("--all", action="store_true", help="Profile all key checkpoints")
    parser.add_argument("--skip_cpu", action="store_true", help="Skip CPU latency benchmarking")
    parser.add_argument("--output_csv", default="results/efficiency.csv", help="CSV output path")
    args = parser.parse_args()

    models_to_profile = []
    if args.all:
        candidates = [
            ("Model 1 Fair (PlainDeepJSCC)", "configs/model1_fair.yaml", "checkpoints/model1_fair/best.pt"),
            ("Model 2d (ResNetJSCC_2d)", "configs/model2d.yaml", "checkpoints/model2d/best.pt"),
            ("Model 2d Long (ResNetJSCC_60ep)", "configs/model2d_long.yaml", "checkpoints/model2d_long/best.pt"),
            ("Model 3 SwinIR (SwinRefineJSCC)", "configs/model3_swinir.yaml", "checkpoints/model3_swinir/best.pt"),
            ("Model 3b lam=0.0003 (Semantic Refiner)", "configs/model3b_lam0.0003.yaml", "checkpoints/model3b_lam0.0003/best.pt"),
            ("Model 4 (Full SwinJSCC)", "configs/model4.yaml", "checkpoints/model4/best.pt"),
            ("Ablation C (SwinJSCC-NoShift)", "configs/model4_noshift.yaml", "checkpoints/model4_noshift/best.pt"),
            ("Ablation A (ConvJSCC Baseline)", "configs/model4_conv.yaml", "checkpoints/model4_conv/best.pt"),
        ]
        for name, cfg_p, ckpt_p in candidates:
            if os.path.isfile(cfg_p) and os.path.isfile(ckpt_p):
                models_to_profile.append((name, cfg_p, ckpt_p))
            else:
                print(f"[SKIP] Missing config or checkpoint for {name}")
    elif args.config and args.checkpoint:
        name = os.path.basename(args.config).replace(".yaml", "")
        models_to_profile.append((name, args.config, args.checkpoint))
    else:
        parser.print_help()
        return

    results = []
    for name, cfg_p, ckpt_p in models_to_profile:
        res = profile_model_entry(name, cfg_p, ckpt_p, profile_cpu=not args.skip_cpu)
        results.append(res)

    if results:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        fieldnames = list(results[0].keys())
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[OK] Efficiency metrics saved to {args.output_csv}")

        # Print Markdown table summary
        print("\n### Model Efficiency Summary Table")
        print("| Model | Total Params | Trainable | Ckpt Size (MB) | GPU Latency (bs=1) | GPU Throughput (bs=64) | Peak VRAM |")
        print("|---|---|---|---|---|---|---|")
        for r in results:
            print(f"| {r['model']} | {r['total_params']:,} | {r['trainable_params']:,} | {r['ckpt_size_mb']} MB | "
                  f"{r['gpu_latency_bs1_ms']} ms/img | {r['gpu_fps_bs64']} img/s | {r['gpu_peak_vram_mb']} MB |")


if __name__ == "__main__":
    main()

