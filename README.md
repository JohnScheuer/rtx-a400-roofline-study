# 🔬 NVIDIA RTX A400 (Ampere SM86) Edge LLM Serving & Roofline Study

[![Hardware](https://img.shields.io/badge/GPU-NVIDIA%20RTX%20A400-green.svg)](https://www.nvidia.com)
[![Architecture](https://img.shields.io/badge/Architecture-Ampere%20SM86-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1%2Bcu121-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Author:** João Felipe de Souza  
**Focus:** ML Systems Engineering | LLM Inference Runtime | Hardware Profiling  
**GitHub:** [github.com/JohnScheuer](https://github.com/JohnScheuer) | **LinkedIn:** [linkedin.com/in/joaofelipescheuer](https://linkedin.com/in/joaofelipescheuer)

---

## 📌 Executive Summary & Systems Insights

This repository contains an empirical, low-level profiling suite executed on an **NVIDIA RTX A400 (Ampere SM86)** under strict edge deployment constraints (**4GB VRAM envelope**, **96 GB/s GDDR6 peak bandwidth**, and **PCIe Gen3 x8 host interconnect**).

Benchmarking `Qwen/Qwen2.5-1.5B-Instruct` across PyTorch 2.5, SDPA backends, WDDM driver limits, and pure CUDA microbenchmarks yielded **5 core architecture takeaways**:

1. **Hardware Ridge Point (115.38 FLOP/Byte):** Pure CUDA microbenchmarks measured an effective bandwidth of **89.30 GB/s** (93% of GDDR6 peak) and **10.30 TFLOPS** in Tensor Cores, deriving a Hardware Ridge Point of **115.38 FLOP/Byte**. Since LLM Decode operates at $I \approx 1.0 \text{ FLOP/Byte}$, decode is strictly **Memory-Bound (>100x below the ridge point)**.
2. **The 32.3x "WDDM Paging Cliff":** Cross-validating PyTorch memory allocation (`torch.cuda.max_reserved`) against physical driver allocation (`nvidia-smi`) proved that contexts $\ge 3072$ tokens trigger transparent Windows WDDM system RAM paging over PCIe Gen3 x8. Oversubscribing VRAM by 2.4 GB causes a **32.3x throughput collapse (6.50 tok/s $\to$ 0.20 tok/s)** without raising a `CUDA OutOfMemoryError`.
3. **PyTorch SDPA Kernel Isolation (18.5x Memory Win):** High-level framework abstractions mask silent kernel fallbacks. Windows PyTorch wheels do not compile FlashAttention natively. Isolating `F.scaled_dot_product_attention` backends directly proved that Cutlass **`EFFICIENT_ATTENTION`** yields a **7x speedup** and an **18.5x attention matrix memory reduction** at 2048 tokens over naive MATH attention (38.1 MB vs 704.1 MB).
4. **Arithmetic Intensity via Dynamic Batching:** Scaling batch size from 1 to 4 increased aggregate server throughput by **3.69x (22.95 tok/s $\to$ 84.61 tok/s)** while increasing per-user latency by only **+8.4% (43.58ms $\to$ 47.27ms)**. Batching converts GEMV to GEMM, multiplying arithmetic intensity without consuming extra VRAM bandwidth.
5. **CUDA Graph Submersion Law:** Capturing CUDA Graphs eliminated ~0.21ms of host driver kernel launch overhead ($T_{dispatch}$). However, because execution is severely memory-bandwidth bound (~46ms spent reading weights per token), CPU launch reduction accounts for only **0.45% of total token time (1.00x speedup)**.

---

## 📊 Target Hardware Specifications

| Parameter | Specification | Impact on Runtime Architecture |
|---|---|---|
| **GPU Architecture** | NVIDIA Ampere (SM86) | 3rd Gen Tensor Cores, Native BF16 & TF32, `cp.async` |
| **Physical VRAM** | 4.00 GB GDDR6 (4093.5 MB) | Restricts context window and KV Cache block capacity |
| **Theoretical Peak Bandwidth** | 96.0 GB/s | Severe decode-phase memory bandwidth bottleneck |
| **Host Interconnect** | PCIe Gen3 x8 (~7.8 GB/s) | Host-to-Device transfer bottleneck during driver paging |
| **Measured Tensor Compute** | 10.30 TFLOPS (FP16/BF16) | Compute headroom remains largely idle during single-user decode |
| **Measured Pure Bandwidth** | 89.30 GB/s (93.0% Peak) | Real-world GDDR6 bus saturation ceiling |

---

## 🧪 Empirical Benchmark Suite

### Test 1: Precision & Bandwidth Saturation (FP16 vs. BF16 Native)
Evaluates whether native BF16 execution on Ampere SM86 improves decode throughput under memory bandwidth starvation.
FP16 Standard : 22.40 tok/s | TPOT: 44.64 ms/tok | Achieved BW: 64.41 GB/s (67.1% Peak)
BF16 Native : 22.36 tok/s | TPOT: 44.72 ms/tok | Achieved BW: 64.30 GB/s (67.0% Peak)

* **Takeaway:** Performance is 100% identical because reading 2.88 GB of model weights (2 bytes/param) dominates execution time. Tensor Core compute capability is irrelevant during single-user decode.

---

### Test 2: Prefill (Compute-Bound) vs. Decode (Memory-Bound) Scaling
Measures Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT) across prompt lengths.

| Prompt Length | TTFT (Prefill Latency) | Prefill Throughput | TPOT (Decode Latency) | Peak VRAM |
|---|---|---|---|---|
| **64 tokens** | 59.09 ms | 1,083.2 tok/s | 43.42 ms/tok (23.0 tok/s) | 2981.1 MB |
| **256 tokens** | 145.55 ms | **1,758.9 tok/s** | 45.16 ms/tok (22.1 tok/s) | 3061.2 MB |
| **512 tokens** | 445.03 ms | 1,150.5 tok/s | 48.68 ms/tok (20.5 tok/s) | 3197.9 MB |
| **1024 tokens** | 1,690.26 ms | 605.8 tok/s | **89.60 ms/tok (11.2 tok/s)** | 3503.8 MB (87.5%) |

* **Takeaway:** Prefill efficiency peaks at **256 tokens (1,758 tok/s)** as GEMM size properly saturates SM86 SMs. At **1024 tokens**, VRAM utilization reaches **87.5%**, causing a **2x decode latency jump** due to contiguous KV cache allocation overhead.

---

### Test 3: Dynamic Batching & Arithmetic Intensity
Evaluates aggregate throughput scaling and per-user latency degradation under concurrent request batches.

| Batch Size | Aggregate Throughput | Speedup | Per-User TPOT Latency | Latency Delta | VRAM BW Used |
|---|---|---|---|---|---|
| **Batch 1** | 22.95 tok/s | 1.00x | 43.58 ms/tok | Baseline | 65.98 GB/s (68.7%) |
| **Batch 2** | 43.77 tok/s | 1.91x | 45.69 ms/tok | +4.8% | 62.93 GB/s (65.6%) |
| **Batch 4** | **84.61 tok/s** | **3.69x** | 47.27 ms/tok | **+8.4%** | 60.82 GB/s (63.4%) |

* **Takeaway:** Batch size 4 yields a **3.69x aggregate throughput gain** while increasing user-perceived latency by only **3.69 ms (+8.4%)**. Batching converts GEMV (memory-bound) into GEMM (compute-bound), raising arithmetic intensity without increasing VRAM bandwidth demand.

---

### Test 4 & 4B: Isolated PyTorch SDPA Kernel Dispatch Analysis
Isolates `F.scaled_dot_product_attention` backends (`FLASH_ATTENTION`, `EFFICIENT_ATTENTION`, `MATH`) on SM86.
UserWarning: Torch was not compiled with flash attention.
FLASH_ATTENTION : UNSUPPORTED (Windows Wheel Build Limitation)

| Sequence Length | MATH Latency | EFFICIENT Latency | Kernel Speedup | MATH Memory | EFFICIENT Memory | Memory Reduction |
|---|---|---|---|---|---|---|
| **512 tokens** | 2.664 ms | **0.383 ms** | **6.95x** | 65.6 MB | **16.1 MB** | **4.0x** |
| **1024 tokens** | 9.117 ms | **1.406 ms** | **6.48x** | 200.1 MB | **23.1 MB** | **8.6x** |
| **2048 tokens** | 37.961 ms | **5.409 ms** | **7.01x** | 704.1 MB | **38.1 MB** | **18.5x** |

* **Takeaway:** Abstractions mask silent fallbacks (`attn_implementation="sdpa"` fell back to MATH on Windows). Explicitly enforcing Cutlass **`EFFICIENT_ATTENTION`** delivers a **7x speedup** and an **18.5x memory reduction** at 2048 tokens (704.1 MB $\to$ 38.1 MB).

---

### Test 5 & 8: WDDM Paging & PCIe Gen3 x8 Oversubscription Telemetry
Cross-validates PyTorch virtual memory allocation (`torch.cuda.max_reserved`) against physical driver allocation (`nvidia-smi`) and PCIe link metrics (`gen=3 / width=x8`).

| Context Length | PyTorch Reserved | `nvidia-smi` Physical Used | Oversubscription | Throughput | Degradation |
|---|---|---|---|---|---|
| **1024 tokens** | 3,226.0 MB (78.8%) | 3,849 MB | **0.0 MB** | **6.506 tok/s** | Baseline (VRAM Resident) |
| **2048 tokens** | 3,778.0 MB (92.3%) | 3,942 MB | **0.0 MB** | **2.256 tok/s** | -65.3% (Physical VRAM Ceiling) |
| **3072 tokens** | **5,000.0 MB** (122.1%) | 3,842 MB | **+906.5 MB** | **0.401 tok/s** | -93.8% (PCIe Paging Active) |
| **4096 tokens** | **6,526.0 MB** (159.4%) | 3,783 MB | **+2,432.5 MB** | **0.201 tok/s** | **-96.9% (32.3x Collapse)** |

* **Takeaway:** PyTorch `mem_get_info()` reports `free=0.0 MB` once physical VRAM saturates (~3.8 GB). Instead of raising an OOM error, the Windows WDDM driver transparently pages 2.4 GB of excess buffers into System RAM over PCIe Gen3 x8. Because PCIe Gen3 x8 (~7.8 GB/s) is **11.4x slower** than GDDR6 (89.3 GB/s), serving throughput suffers a **32.3x collapse**.

---

### Test 6: CUDA Graph Replay vs. Eager CPU Dispatch Submersion
Quantifies CPU launch overhead ($T_{dispatch}$) reduction using PyTorch static CUDA Graph capture (`reduce-overhead`).
Eager Mode (Host Kernel Launches) : 46.39 ms/tok | 21.6 tok/s | 1.00x
CUDA Graph (Static Graph Replay) : 46.18 ms/tok | 21.7 tok/s | 1.00x (+0.45% Delta)
* **Takeaway (The Submersion Law):** CUDA Graphs saved **~0.21 ms per token** of host driver launch overhead. However, because decode is heavily memory-bandwidth bound (~46ms spent reading weights), CPU launch reduction accounts for only **0.45% of total time**. CUDA Graphs deliver massive speedups in Launch-Bound/Compute-Bound regimes, but their benefit is completely submerged in severely memory-bound edge environments.

---

### Test 7: Pure Hardware Microbenchmark & Ridge Point Derivation
Measures pure physical hardware limits without framework/HuggingFace overhead.

$$\text{Hardware Ridge Point} = \frac{10.30 \text{ TFLOPS}}{89.30 \text{ GB/s}} = \mathbf{115.38 \text{ FLOP/Byte}}$$
Measured VRAM Bandwidth : 89.30 GB/s (93.0% of theoretical peak)
FP32 CUDA Cores Peak : 1.82 TFLOPS
FP16 / BF16 Tensor Cores Peak : 10.30 TFLOPS (5.66x Tensor Core acceleration)
* **Takeaway:** To fully saturate SM86 Tensor Cores (10.3 TFLOPS), a kernel must achieve an Operational Intensity $\ge 115.38 \text{ FLOP/Byte}$. LLM Decode operates at $I \approx 1.0 \text{ FLOP/Byte}$, sitting **>100x below the ridge point**. This mathematically proves why LLM Decode is strictly Memory-Bound on edge hardware.

---

## 📁 Repository Structure
.
├── a400-ampere-roofline-study.md # Complete technical whitepaper report
├── benchmark_a400.py # Test 1: Precision & Bandwidth Saturation
├── benchmark_prefill_a400.py # Test 2: Prefill vs Decode & Context Scaling
├── benchmark_batch_a400.py # Test 3: Dynamic Batching & Arithmetic Intensity
├── benchmark_flashattn_a400.py # Test 4: High-Level SDPA vs Eager Model Test
├── benchmark_sdpa_backend.py # Test 4B: Isolated PyTorch SDPA Kernel Profiling
├── benchmark_oom_frontier_a400.py # Test 5: Maximum Context Frontier Test
├── benchmark_cudagraph_a400.py # Test 6: CUDA Graph vs Eager Dispatch Test
├── benchmark_hardware_roofline_a400.py # Test 7: Pure CUDA Hardware Microbenchmark
├── benchmark_wddm_paging_telemetry_a400.py # Test 8: Cross-Validated WDDM Paging Telemetry
├── wddm_paging_telemetry_results.json # Raw telemetry output dataset
├── nvidia-smi.txt # Physical GPU environment dump
└── software_versions.txt # PyTorch / CUDA / Transformers toolchain versions

---

## 🛠️ How to Reproduce

### 1. Prerequisites
- **Hardware:** NVIDIA GPU (Tested on RTX A400 SM86 4GB).
- **OS:** Windows 10/11 or Linux Ubuntu 22.04.
- **Python:** 3.11+
- **CUDA:** 12.1+

### 2. Environment Setup
```powershell
# Install PyTorch with CUDA 12.1 support
pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# Install HuggingFace dependencies
pip install transformers accelerate

3. Run Benchmarks
# Run pure hardware roofline microbenchmark
python benchmark_hardware_roofline_a400.py

# Run cross-validated WDDM paging telemetry test
python benchmark_wddm_paging_telemetry_a400.py

# Run isolated SDPA kernel profiling
python benchmark_sdpa_backend.py

👤 Author & Contact
João Felipe de Souza
Senior ML Systems Engineer | LLM Inference Runtime & GPU Performance

GitHub: github.com/JohnScheuer
LinkedIn: linkedin.com/in/joaofelipescheuer
Email: johnfelipe13@gmail.com
WeChat: JohnScheuer7




