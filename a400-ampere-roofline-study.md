# RTX A400 (Ampere SM86) LLM Inference Profiling & Roofline Analysis
**Author:** João Felipe de Souza  
**Hardware Target:** NVIDIA RTX A400 (4GB VRAM, 96 GB/s Peak Bandwidth, Compute Capability 8.6)  
**Model Benchmarked:** `Qwen/Qwen2.5-1.5B-Instruct` (FP16 / BF16)  

---

## Executive Summary
This document summarizes empirical profiling data collected on an **NVIDIA RTX A400 (Ampere SM86)** under strict edge constraints (4GB VRAM envelope and a narrow memory bandwidth ceiling of 96 GB/s). The study evaluates memory bandwidth saturation, prefill vs. decode scaling, and dynamic batching efficiency to model real-world serving bottlenecks.

---

## 1. Precision & Bandwidth Saturation (FP16 vs. BF16)
* **Goal:** Determine if native BF16 support on Ampere changes decode throughput compared to FP16 under severe memory bandwidth starvation.
* **Findings:**
  * **FP16 Standard:** 22.40 tok/s | Achieved BW: 64.41 GB/s (**67.1% of Roofline Peak**)
  * **BF16 Native:** 22.36 tok/s | Achieved BW: 64.30 GB/s (**67.0% of Roofline Peak**)
* **Engineering Takeaway:** Performance is identical because model weight transfer dominates the decode phase. FLOP compute capacity sits idle waiting for VRAM reads, proving that memory bandwidth is the absolute bottleneck in edge inference.

---

## 2. Prefill vs. Decode & The 1024-Token "Memory Cliff"
* **Goal:** Measure TTFT (Compute-Bound Prefill) against TPOT (Memory-Bound Decode) across scaling prompt lengths.
* **Results Matrix:**

| Prompt Length | TTFT (Prefill Latency) | TPOT (Decode Latency) | Peak VRAM Allocated |
|---|---|---|---|
| **64 tokens** | 59.09 ms (1,083 tok/s) | 43.42 ms/tok (23.0 t/s) | 2981.1 MB |
| **256 tokens** | 145.55 ms (1,758 tok/s) | 45.16 ms/tok (22.1 t/s) | 3061.2 MB |
| **512 tokens** | 445.03 ms (1,150 tok/s) | 48.68 ms/tok (20.5 t/s) | 3197.9 MB |
| **1024 tokens** | 1,690.26 ms (605 tok/s) | 89.60 ms/tok (11.2 t/s) | 3503.8 MB |

* **Engineering Takeaway:** 
  * Prefill efficiency peaks at **256 tokens** as GEMM matrix sizes properly saturate the SM86 Tensor Cores.
  * At **1024 tokens**, VRAM consumption hits **87.5% capacity (3.5GB)**, triggering a severe **Memory Cliff**: decode latency doubles (89.6 ms/tok) due to contiguous KV Cache allocation overhead and fragmentation. This proves that PagedAttention is mandatory for $\le 4\text{GB}$ devices.

---

## 3. Dynamic Batching & Arithmetic Intensity Scaling
* **Goal:** Evaluate aggregate throughput scaling and per-user latency impact when increasing concurrency.
* **Results Matrix:**

| Batch Size | Aggregate Throughput | Per-User TPOT Latency | VRAM Peak |
|---|---|---|---|
| **Batch 1** | 22.95 tok/s (1.00x) | 43.58 ms/tok | 2961.5 MB |
| **Batch 2** | 43.77 tok/s (1.91x) | 45.69 ms/tok | 2972.8 MB |
| **Batch 4** | 84.61 tok/s (3.69x) | 47.27 ms/tok | 2991.0 MB |

* **Engineering Takeaway:** Scaling batch size from 1 to 4 yields a **3.69x increase in aggregate server throughput** while increasing per-user latency by only **+8.4%**. Batching successfully transitions execution from memory-bound GEMV to arithmetic-heavy GEMM without consuming extra VRAM bandwidth.