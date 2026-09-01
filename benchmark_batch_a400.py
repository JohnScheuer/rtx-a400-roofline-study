import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PEAK_BANDWIDTH_GBS = 96.0

def print_header():
    print("=" * 80)
    print("   RTX A400 (SM86) BATCH SIZE SCALING & ARITHMETIC INTENSITY STUDY   ")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80 + "\n")

def run_batch_study():
    print_header()
    
    print("--> Loading model into VRAM (BF16 Native)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_gb = model_bytes / (1024**3)
    
    # Batch sizes a testar (1, 2, 4) -> 8 estoura os 4GB VRAM
    batch_sizes = [1, 2, 4]
    gen_tokens = 64
    single_prompt = "Explain why batching increases arithmetic intensity in GPU matrix multiplication."
    
    results = []
    
    print("\nStarting Batch Scaling Matrix...\n")
    
    for b_size in batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Criar batch com b_size cópias do prompt
        prompts = [single_prompt] * b_size
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        input_len = inputs.input_ids.shape[1]
        
        # 1. Warmup
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=5)
        torch.cuda.synchronize()
        
        # 2. Medir Geração em Batch
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=gen_tokens, 
                min_new_tokens=gen_tokens,
                do_sample=False
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        total_time = t1 - t0
        total_gen_tokens = b_size * gen_tokens
        
        # Métricas Chave
        aggregate_tok_per_sec = total_gen_tokens / total_time
        per_user_tpot_ms = (total_time / gen_tokens) * 1000.0  # Latência vista por 1 usuário no batch
        
        # Reutilização de Banda:
        # No decode de batch B, para cada passo de tempo lemos 'model_gb' 1 vez e geramos B tokens.
        effective_read_gb = model_gb * gen_tokens # Leitura real da VRAM
        achieved_bw_gbs = (effective_read_gb / total_time) # Banda física consumida
        
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        
        print(f" Batch Size: {b_size:<2} requests | Total Tokens Generated: {total_gen_tokens} tokens")
        print(f"  • Aggregate Throughput:      {aggregate_tok_per_sec:>7.2f} tok/s  (Vazão total do servidor)")
        print(f"  • Per-User TPOT Latency:     {per_user_tpot_ms:>7.2f} ms/tok (Latência por token por usuário)")
        print(f"  • Physical VRAM BW Used:     {achieved_bw_gbs:>7.2f} GB/s   ({(achieved_bw_gbs/PEAK_BANDWIDTH_GBS)*100:.1f}% peak)")
        print(f"  • Peak VRAM Allocated:       {peak_vram_mb:>7.1f} MB")
        print("-" * 80)
        
        results.append({
            "batch_size": b_size,
            "agg_tok_s": aggregate_tok_per_sec,
            "per_user_tpot": per_user_tpot_ms,
            "vram_bw": achieved_bw_gbs,
            "peak_vram": peak_vram_mb
        })
        
    # Exibir Tabela Consolidada
    print("\n" + "=" * 80)
    print("                     BATCH SIZE SCALING SUMMARY MATRIX                   ")
    print("=" * 80)
    print(f"{'Batch Size':<12} | {'Aggregate Tok/s':<18} | {'Per-User TPOT (ms)':<20} | {'VRAM Peak':<10}")
    print("-" * 80)
    b1_agg = results[0]['agg_tok_s']
    for r in results:
        speedup = r['agg_tok_s'] / b1_agg
        print(f"Batch {r['batch_size']:<6} | {r['agg_tok_s']:>7.2f} tok/s ({speedup:.2f}x) | {r['per_user_tpot']:>8.2f} ms/tok         | {r['peak_vram']:>6.1f} MB")
    print("=" * 80)

if __name__ == "__main__":
    run_batch_study()