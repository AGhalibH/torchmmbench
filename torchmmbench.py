#!/usr/bin/env python3
import argparse
import time
import torch
import platform


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch MatMul Benchmark. Simple PyTorch Benchmark to Calculate CPU/GPU (NVIDIA/AMD/INTEL ARC) Throughput Performance (TFLOPS) using Matrix Multiplication"
    )
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "xpu"],
                        help="Device to run on (default: cpu)")
    parser.add_argument("--size", type=int, default=8192,
                        help="Matrix size (NxN) (default: 8192)")
    parser.add_argument("--iters", type=int, default=10,
                        help="Number of benchmark iterations (default: 10)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations (default: 3)")
    parser.add_argument("--threads", type=int, default=None,
                        help="Number of CPU threads (only applies to cpu device)")
    return parser.parse_args()


def get_device(device_str):
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return torch.device("cuda")
    elif device_str == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU not available")
        return torch.device("xpu")
    else:
        return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu_memory = props.total_memory / (1024**3)
        arch = getattr(props, "gcnArchName", None)
        if arch:
            return f"{props.name} ({gpu_memory:.1f} GB) | {arch}"
        return f"{props.name} ({gpu_memory:.1f} GB) | CC {props.major}.{props.minor}"

    elif device.type == "xpu":
        if hasattr(torch, "xpu"):
            props = torch.xpu.get_device_properties(device)
            gpu_memory = props.total_memory / (1024**3)
            return f"{props.name} ({gpu_memory:.1f} GB)"
        return "XPU (unsupported build)"

    elif device.type == "cpu":
        cpu_name = platform.processor()
        return cpu_name if cpu_name else "CPU"

    return f"Unknown device ({device})"


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


def benchmark_dtype(device, dtype, N, iters, warmup):
    # Allocate tensors
    A = torch.randn((N, N), device=device, dtype=dtype)
    B = torch.randn((N, N), device=device, dtype=dtype)

    # Warm-up
    for _ in range(warmup):
        _ = torch.matmul(A, B)
    synchronize(device)

    # Benchmark
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = torch.matmul(A, B)
        synchronize(device)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)

    # FLOPs
    flops = 2 * (N ** 3)
    tflops = flops / avg_time / 1e12

    return avg_time * 1000, tflops  # ms, TFLOPS


def main():
    args = parse_args()

    device = get_device(args.device)

    # CPU tuning
    if device.type == "cpu":
        if args.threads is not None:
            torch.set_num_threads(args.threads)

    print("=" * 55)
    print("PyTorch MatMul (Matrix Multiplication) Benchmark")
    print("=" * 55)

    print("Configuration:")
    print(f"  OS          : {platform.system()}")
    print(f"  Device      : {device}")
    print(f"  Device Name : {get_device_name(device)}")
    print(f"  PyTorch Ver : {torch.__version__}")
    print(f"  Matrix Size : {args.size} x {args.size}")
    print(f"  Iterations  : {args.iters}")
    print(f"  Warmup      : {args.warmup}")
    if device.type == "cpu" and args.threads is not None:
        print(f"  Threads     : {args.threads}")
    print("=" * 55)

    # Select dtypes
    dtype_list = [torch.float32]

    if device.type in ["cuda", "xpu"]:
        dtype_list.append(torch.float16)
        bf16_supported = False
        if device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported"):
            bf16_supported = torch.cuda.is_bf16_supported()
        elif device.type == "xpu" and hasattr(torch.xpu, "is_bf16_supported"):
            bf16_supported = torch.xpu.is_bf16_supported()
        if bf16_supported:
            dtype_list.append(torch.bfloat16)

    results = []

    for dtype in dtype_list:
        try:
            avg_ms, tflops = benchmark_dtype(
                device, dtype, args.size, args.iters, args.warmup
            )
            results.append((str(dtype).replace("torch.", ""), avg_ms, tflops))
        except Exception as e:
            results.append((str(dtype).replace("torch.", ""), None, None))

    # Print table
    print("\nResults:")
    print("=" * 55)
    print(f"| {'DTYPE':<10} | {'AVG TIME (ms)':<15} | {'THROUGHPUT (TFLOPS)':<20} |")
    print("=" * 55)

    for dtype, avg_ms, tflops in results:
        if avg_ms is None:
            print(f"| {dtype:<10} | {'ERROR':<15} | {'ERROR':<20} |")
        else:
            print(f"| {dtype:<10} | {avg_ms:<15.2f} | {tflops:<20.2f} |")

    print("=" * 55)
    print("Benchmark Completed\n")


if __name__ == "__main__":
    main()
