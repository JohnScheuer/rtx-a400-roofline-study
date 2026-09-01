import time
import torch

def print_header():
    print("=" * 80)
    print("   RTX A400 (SM86) RAW HARDWARE ROOFLINE MICROBENCHMARK (PURE CUDA/PYTORCH)   ")
    print("=" * 80)
    dev_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Device:               {dev_name}")
    print(f"Architecture:         SM{cap[0]}.{cap[1]} ({'Ampere' if cap[0]==8 else 'Other'})")
    print(f"Physical VRAM:        {vram_gb:.2f} GB")
    print("=" * 80 + "\n")

def benchmark_memory_bandwidth():
    print("--> 1. Measuring Pure VRAM Read/Write Bandwidth (GB/s)...")
    # Alocar dois vetores grandes de 500MB cada (1GB VRAM total)
    num_elements = 125 * 1024 * 1024  # 125M float32 elements = 500 MB
    a = torch.randn(num_elements, device="cuda", dtype=torch.float32)
    b = torch.empty_like(a)
    
    # Warmup
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    
    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    total_time = (t1 - t0) / iters
    bytes_transferred = num_elements * 4 * 2  # Read 'a' (4 bytes) + Write 'b' (4 bytes)
    achieved_bw_gbs = (bytes_transferred / 1e9) / total_time
    
    print(f"    - Measured VRAM Bandwidth: {achieved_bw_gbs:.2f} GB/s (Peak GDDR6 limit)\n")
    return achieved_bw_gbs

def benchmark_gemm_flops(dtype, name):
    print(f"--> Measuring [{name}] Tensor Compute Performance...")
    # Matriz grande 8192 x 8192 para lotar os SMs do Ampere
    N = 8192
    a = torch.randn(N, N, device="cuda", dtype=dtype)
    b = torch.randn(N, N, device="cuda", dtype=dtype)
    
    # Warmup
    for _ in range(5):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    avg_time = (t1 - t0) / iters
    # FLOPs para GEMM N x N = 2 * N^3
    total_flops = 2 * (N ** 3)
    tflops = (total_flops / 1e12) / avg_time
    
    print(f"    - {name:<25}: {tflops:.2f} TFLOPS (Execution Time: {avg_time*1000:.2f} ms)\n")
    return tflops

def main():
    print_header()
    
    # 1. Bandwidth
    real_bw = benchmark_memory_bandwidth()
    
    # 2. FP32 CUDA Cores
    fp32_tflops = benchmark_gemm_flops(torch.float32, "FP32 (CUDA Cores)")
    
    # 3. FP16 Tensor Cores
    fp16_tflops = benchmark_gemm_flops(torch.float16, "FP16 (Tensor Cores)")
    
    # 4. BF16 Tensor Cores
    bf16_tflops = benchmark_gemm_flops(torch.bfloat16, "BF16 (Tensor Cores)")
    
    # Cálculo do Ridge Point (Hardware Inflexion Point)
    ridge_fp16 = (fp16_tflops * 1e12) / (real_bw * 1e9)
    ridge_bf16 = (bf16_tflops * 1e12) / (real_bw * 1e9)
    
    print("=" * 80)
    print("              RTX A400 HARDWARE ROOFLINE CHARACTERISTICS MATRIX             ")
    print("=" * 80)
    print(f"  Measured VRAM Bandwidth:       {real_bw:.2f} GB/s")
    print(f"  FP32 CUDA Cores Peak:          {fp32_tflops:.2f} TFLOPS")
    print(f"  FP16 Tensor Cores Peak:        {fp16_tflops:.2f} TFLOPS")
    print(f"  BF16 Tensor Cores Peak:        {bf16_tflops:.2f} TFLOPS")
    print(f"  Tensor Core vs CUDA Core Boost: {fp16_tflops / fp32_tflops:.2f}x Compute Acceleration")
    print("-" * 80)
    print(f"  Hardware Ridge Point (FP16):   {ridge_fp16:.2f} FLOP/Byte")
    print(f"  Hardware Ridge Point (BF16):   {ridge_bf16:.2f} FLOP/Byte")
    print("=" * 80)
    print("\n* INTERPRETATION:")
    print(f"  Any kernel with Operational Intensity < {ridge_fp16:.1f} FLOP/Byte is MEMORY-BOUND on RTX A400.")
    print(f"  LLM Decode phase has Operational Intensity ~1.0 FLOP/Byte, which is FAR below {ridge_fp16:.1f}.")
    print("  This mathematically proves why LLM Decode is strictly Memory-Bound on this hardware.")
    print("=" * 80)

if __name__ == "__main__":
    main()