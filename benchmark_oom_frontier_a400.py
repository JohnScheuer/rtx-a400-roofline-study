import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TOTAL_GPU_VRAM_MB = torch.cuda.get_device_properties(0).total_memory / (1024**2)

def print_header():
    print("=" * 80)
    print("   RTX A400 (SM86) MAXIMUM CONTEXT FRONTIER & KV CACHE COST PROFILING   ")
    print("=" * 80)
    print(f"Device:               {torch.cuda.get_device_name(0)}")
    print(f"Physical VRAM Limit:  {TOTAL_GPU_VRAM_MB:.1f} MB (4.00 GB)")
    print("=" * 80 + "\n")

def run_oom_frontier_study():
    print_header()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Carregar modelo em BF16 com SDPA
    print("--> Loading model in BF16 Native + SDPA...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="sdpa",
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    
    base_model_vram_mb = torch.cuda.memory_allocated() / (1024**2)
    print(f"--> Base Model Static Weight Size: {base_model_vram_mb:.1f} MB\n")
    
    # Varredura de contexto: 1024 até 6144 tokens
    context_lengths = [1024, 2048, 3072, 4096, 5120, 6144]
    gen_tokens = 32
    base_text = "NVIDIA CUDA GPU memory allocation KV cache context window limit " * 500
    
    results = []
    
    for ctx_len in context_lengths:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        tokens = tokenizer(base_text, return_tensors="pt", max_length=ctx_len, truncation=True).to("cuda")
        actual_len = tokens.input_ids.shape[1]
        
        print(f"--> Testing Context Length: {actual_len} tokens...")
        
        try:
            # Warmup
            with torch.inference_mode():
                _ = model.generate(**tokens, max_new_tokens=2)
            torch.cuda.synchronize()
            
            # Test Run
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(
                    **tokens, 
                    max_new_tokens=gen_tokens, 
                    min_new_tokens=gen_tokens,
                    do_sample=False
                )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            total_time = t1 - t0
            tok_per_sec = gen_tokens / total_time
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
            vram_headroom_mb = TOTAL_GPU_VRAM_MB - peak_vram_mb
            vram_util_pct = (peak_vram_mb / TOTAL_GPU_VRAM_MB) * 100.0
            
            # Estimativa empírica de VRAM usada por token de contexto
            kv_estimated_mb = peak_vram_mb - base_model_vram_mb
            mb_per_token = kv_estimated_mb / actual_len if actual_len > 0 else 0
            
            print(f"    [SUCCESS]")
            print(f"    - Execution Time:      {total_time:.2f} s ({tok_per_sec:.1f} tok/s)")
            print(f"    - Peak VRAM Allocated: {peak_vram_mb:.1f} MB / {TOTAL_GPU_VRAM_MB:.1f} MB ({vram_util_pct:.1f}%)")
            print(f"    - VRAM Headroom Left:  {vram_headroom_mb:.1f} MB")
            print(f"    - Empirical Cost:      ~{mb_per_token:.4f} MB/token")
            print("-" * 80)
            
            results.append({
                "context_len": actual_len,
                "status": "SUCCESS",
                "peak_vram": peak_vram_mb,
                "headroom": vram_headroom_mb,
                "util_pct": vram_util_pct,
                "mb_per_token": mb_per_token,
                "tok_s": tok_per_sec
            })
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"    [HARD OOM ENCOUNTERED]")
            print(f"    - CUDA OutOfMemory at {actual_len} tokens!")
            print("-" * 80)
            results.append({
                "context_len": actual_len,
                "status": "OOM (Out Of Memory)",
                "peak_vram": TOTAL_GPU_VRAM_MB,
                "headroom": 0,
                "util_pct": 100.0,
                "mb_per_token": 0,
                "tok_s": 0
            })
            break # Parar na primeira falha de OOM
        except Exception as e:
            print(f"    [FAILED]: {e}")
            break

    # Exibir Tabela Consolidada
    print("\n" + "=" * 80)
    print("                4GB VRAM MAXIMUM CONTEXT FRONTIER MATRIX                ")
    print("=" * 80)
    print(f"{'Context Len':<12} | {'Status':<10} | {'Peak VRAM':<14} | {'Headroom':<12} | {'Cost/Token':<12}")
    print("-" * 80)
    for r in results:
        if r['status'] == "SUCCESS":
            print(f"{r['context_len']:<12} | {r['status']:<10} | {r['peak_vram']:>7.1f} MB ({r['util_pct']:.1f}%) | {r['headroom']:>7.1f} MB   | {r['mb_per_token']:.4f} MB/tok")
        else:
            print(f"{r['context_len']:<12} | {r['status']:<10} | {TOTAL_GPU_VRAM_MB:>7.1f} MB (100%) | 0.0 MB       | N/A (OOM)")
    print("=" * 80)

if __name__ == "__main__":
    run_oom_frontier_study()