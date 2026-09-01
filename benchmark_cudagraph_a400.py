import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def print_header():
    print("=" * 80)
    print("   RTX A400 (SM86) CUDA GRAPH REPLAY vs EAGER DISPATCH OVERHEAD STUDY   ")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80 + "\n")

def run_cudagraph_study():
    print_header()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print("--> Loading model into VRAM (BF16 Native)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="sdpa",
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    
    prompt = "Explain CUDA Graph execution vs Eager mode kernel launches in GPU acceleration."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    gen_tokens = 100
    
    print("\n1. Benchmarking Eager Mode Execution...")
    # Warmup
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=5)
    torch.cuda.synchronize()
    
    t0_eager = time.perf_counter()
    with torch.inference_mode():
        out_eager = model.generate(**inputs, max_new_tokens=gen_tokens, min_new_tokens=gen_tokens, do_sample=False)
    torch.cuda.synchronize()
    t1_eager = time.perf_counter()
    
    eager_time = t1_eager - t0_eager
    eager_tok_s = gen_tokens / eager_time
    eager_tpot = (eager_time / gen_tokens) * 1000.0
    
    print(f"   • Eager Mode Total Time:   {eager_time:.3f} s")
    print(f"   • Eager Mode TPOT Latency: {eager_tpot:.2f} ms/tok ({eager_tok_s:.1f} tok/s)")
    
    print("\n2. Benchmarking PyTorch Static Capture / CUDAGraph Integration...")
    # Usar torch.compile / CUDAGraphs do PyTorch para compilar o decode step
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead", backend="inductor")
        
        # Warmup do grafo (compile + capture phase)
        print("   --> Capturing CUDA Graph (Warmup/Compilation phase)...")
        with torch.inference_mode():
            _ = compiled_model.generate(**inputs, max_new_tokens=5)
        torch.cuda.synchronize()
        
        t0_graph = time.perf_counter()
        with torch.inference_mode():
            out_graph = compiled_model.generate(**inputs, max_new_tokens=gen_tokens, min_new_tokens=gen_tokens, do_sample=False)
        torch.cuda.synchronize()
        t1_graph = time.perf_counter()
        
        graph_time = t1_graph - t0_graph
        graph_tok_s = gen_tokens / graph_time
        graph_tpot = (graph_time / gen_tokens) * 1000.0
        
        speedup = eager_time / graph_time
        latency_reduction_ms = eager_tpot - graph_tpot
        
        print(f"   • CUDA Graph Total Time:   {graph_time:.3f} s")
        print(f"   • CUDA Graph TPOT Latency: {graph_tpot:.2f} ms/tok ({graph_tok_s:.1f} tok/s)")
        print(f"   • Measured Speedup:        {speedup:.2f}x faster")
        print(f"   • CPU Dispatch Saved:      ~{latency_reduction_ms:.2f} ms per token!")
        
        status = "SUCCESS"
    except Exception as e:
        print(f"   • CUDA Graph Capture Skipped/Failed: {e}")
        status = "FAILED"
        speedup = 1.0
        graph_tpot = eager_tpot
        graph_tok_s = eager_tok_s
        
    print("\n" + "=" * 80)
    print("                   CUDA GRAPH vs EAGER DISPATCH SUMMARY                   ")
    print("=" * 80)
    print(f"{'Execution Mode':<25} | {'TPOT (ms/tok)':<15} | {'Throughput':<15} | {'Speedup':<10}")
    print("-" * 80)
    print(f"{'Eager Mode (Host Launch)':<25} | {eager_tpot:>8.2f} ms      | {eager_tok_s:>8.1f} tok/s   | 1.00x")
    if status == "SUCCESS":
        print(f"{'CUDA Graph (Replay Exec)':<25} | {graph_tpot:>8.2f} ms      | {graph_tok_s:>8.1f} tok/s   | {speedup:.2f}x")
    print("=" * 80)

if __name__ == "__main__":
    run_cudagraph_study()