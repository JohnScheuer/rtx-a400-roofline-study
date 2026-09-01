import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def print_header():
    print("=" * 75)
    print("   RTX A400 (SM86) PREFILL (COMPUTE-BOUND) vs DECODE (MEMORY-BOUND) STUDY  ")
    print("=" * 75)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 75 + "\n")

def run_prefill_study():
    print_header()
    
    print("--> Loading model into VRAM (BF16 Native)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda",
        low_cpu_mem_usage=True
    )
    
    # Prompt lengths a testar (64 -> 256 -> 512 -> 1024 tokens)
    prompt_lengths = [64, 256, 512, 1024]
    gen_tokens = 64 # tokens a gerar no decode
    
    results = []
    
    # Base synthetic prompt builder
    base_text = "NVIDIA CUDA GPU architecture memory bandwidth Tensor Core optimization " * 200
    
    print("\nStarting Prompt Scaling Matrix...\n")
    
    for length in prompt_lengths:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Tokenizar prompt com o tamanho exato desejado
        tokens = tokenizer(base_text, return_tensors="pt", max_length=length, truncation=True).to("cuda")
        actual_input_len = tokens.input_ids.shape[1]
        
        # 1. Warmup
        with torch.inference_mode():
            _ = model.generate(**tokens, max_new_tokens=2)
        torch.cuda.synchronize()
        
        # 2. Medir TTFT (Prefill Phase - Compute Bound)
        t_prefill_start = time.perf_counter()
        with torch.inference_mode():
            out_prefill = model(**tokens) # Forward pass apenas do prompt
        torch.cuda.synchronize()
        t_prefill_end = time.perf_counter()
        
        ttft_ms = (t_prefill_end - t_prefill_start) * 1000.0
        prefill_tok_per_sec = actual_input_len / (t_prefill_end - t_prefill_start)
        
        # 3. Medir Total Generation (Prefill + Decode)
        t_gen_start = time.perf_counter()
        with torch.inference_mode():
            out_full = model.generate(
                **tokens, 
                max_new_tokens=gen_tokens, 
                min_new_tokens=gen_tokens,
                do_sample=False
            )
        torch.cuda.synchronize()
        t_gen_end = time.perf_counter()
        
        total_time = t_gen_end - t_gen_start
        actual_gen_len = out_full.shape[1] - actual_input_len
        
        # Isolation do Decode Phase
        # T_decode_total = Total_Time - T_prefill
        decode_time_sec = total_time - (ttft_ms / 1000.0)
        tpot_ms = (decode_time_sec / actual_gen_len) * 1000.0
        decode_tok_per_sec = actual_gen_len / decode_time_sec
        
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
        
        print(f" Prompt Length: {actual_input_len:<4} tokens | Gen: {actual_gen_len} tokens")
        print(f"  • TTFT (Prefill Latency):   {ttft_ms:>7.2f} ms  ({prefill_tok_per_sec:>7.1f} tok/s - Compute-Bound)")
        print(f"  • TPOT (Decode Latency):    {tpot_ms:>7.2f} ms/tok ({decode_tok_per_sec:>5.1f} tok/s - Memory-Bound)")
        print(f"  • Peak VRAM Allocated:      {peak_vram_mb:>7.1f} MB")
        print("-" * 75)
        
        results.append({
            "prompt_len": actual_input_len,
            "ttft_ms": ttft_ms,
            "prefill_tok_s": prefill_tok_per_sec,
            "tpot_ms": tpot_ms,
            "decode_tok_s": decode_tok_per_sec,
            "peak_vram": peak_vram_mb
        })
        
    # Exibir Tabela Consolidada
    print("\n" + "=" * 75)
    print("                     PREFILL vs DECODE SCALING MATRIX                   ")
    print("=" * 75)
    print(f"{'Prompt Len':<12} | {'TTFT (Prefill)':<18} | {'TPOT (Decode)':<18} | {'VRAM Peak':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['prompt_len']:<12} | {r['ttft_ms']:>7.2f} ms ({r['prefill_tok_s']:>5.0f} t/s) | {r['tpot_ms']:>7.2f} ms ({r['decode_tok_s']:>4.1f} t/s) | {r['peak_vram']:>6.1f} MB")
    print("=" * 75)

if __name__ == "__main__":
    run_prefill_study()