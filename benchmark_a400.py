import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configurações
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # Modelo de 1.5B (~3GB em FP16, perfeito para 4GB)
PEAK_BANDWIDTH_GBS = 96.0                 # Roofline da RTX A400 GDDR6
PROMPT = "Explain the architecture of NVIDIA Tensor Cores and why memory bandwidth limits LLM decode phase in 200 words."
GEN_TOKENS = 128

def print_header():
    print("=" * 70)
    print("      RTX A400 (AMPERE SM86) MEMORY BANDWIDTH & RUNTIME BENCHMARK     ")
    print("=" * 70)
    dev_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    bf16_ok = torch.cuda.is_bf16_supported()
    print(f"Device:               {dev_name}")
    print(f"Compute Capability:   SM{cap[0]}.{cap[1]} ({'Ampere' if cap[0]==8 else 'Other'})")
    print(f"Total VRAM:           {total_mem:.2f} GB")
    print(f"Native BF16 Support:  {bf16_ok}")
    print(f"Theoretical Peak BW:  {PEAK_BANDWIDTH_GBS} GB/s")
    print("=" * 70 + "\n")

def run_test(dtype, name):
    print(f"--> Loading model in [{name}]...")
    gc.collect()
    torch.cuda.empty_cache()
    
    start_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=dtype, 
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    load_time = time.perf_counter() - start_load
    
    # Calcular tamanho real do modelo na VRAM
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_gb = model_bytes / (1024**3)
    
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")
    input_tokens = inputs.input_ids.shape[1]
    
    # 1. Warmup
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=10)
    torch.cuda.synchronize()
    
    # 2. Benchmark Execute
    print(f"--> Running inference generation ({GEN_TOKENS} tokens)...")
    torch.cuda.reset_peak_memory_stats()
    
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=GEN_TOKENS, 
            min_new_tokens=GEN_TOKENS,
            do_sample=False
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    gen_tokens = outputs.shape[1] - input_tokens
    tok_per_sec = gen_tokens / total_time
    tpot_ms = (total_time / gen_tokens) * 1000  # Time Per Output Token
    
    # Cálculo de Largura de Banda Efetiva Achieved (Memory Bandwidth Roofline)
    # No decode phase, para cada token gerado, a GPU lê TODOS os pesos do modelo 1 vez da VRAM.
    achieved_bw_gbs = model_gb * tok_per_sec
    roofline_eff_pct = (achieved_bw_gbs / PEAK_BANDWIDTH_GBS) * 100.0
    
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    
    print(f"    [Result {name}]")
    print(f"    - Load Time:            {load_time:.2f} s")
    print(f"    - Model Size in VRAM:   {model_gb:.2f} GB")
    print(f"    - Peak VRAM Used:       {peak_vram_mb:.1f} MB")
    print(f"    - Throughput:           {tok_per_sec:.2f} tok/s")
    print(f"    - TPOT (Latency/token): {tpot_ms:.2f} ms/tok")
    print(f"    - Achieved Memory BW:   {achieved_bw_gbs:.2f} GB/s")
    print(f"    - Bandwidth Efficiency: {roofline_eff_pct:.1f}% of Roofline Peak ({PEAK_BANDWIDTH_GBS} GB/s)")
    print("-" * 70 + "\n")
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "name": name,
        "tok_per_sec": tok_per_sec,
        "tpot_ms": tpot_ms,
        "achieved_bw_gbs": achieved_bw_gbs,
        "efficiency_pct": roofline_eff_pct,
        "peak_vram_mb": peak_vram_mb
    }

def main():
    print_header()
    results = []
    
    # Test 1: FP16 Standard
    try:
        r1 = run_test(torch.float16, "FP16 Standard")
        results.append(r1)
    except Exception as e:
        print(f"FP16 Failed: {e}\n")
        
    # Test 2: BF16 Native (Ampere Feature - Não roda nativo no SM75!)
    try:
        r2 = run_test(torch.bfloat16, "BF16 Native (Ampere Feature)")
        results.append(r2)
    except Exception as e:
        print(f"BF16 Failed: {e}\n")

    # Resumo Final em Tabela
    print("=" * 70)
    print("                      FINAL SUMMARY MATRIX                        ")
    print("=" * 70)
    print(f"{'Precision':<25} | {'Tok/s':<10} | {'TPOT (ms)':<10} | {'BW (GB/s)':<10} | {'Efficiency':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} | {r['tok_per_sec']:<10.2f} | {r['tpot_ms']:<10.2f} | {r['achieved_bw_gbs']:<10.2f} | {r['efficiency_pct']:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()