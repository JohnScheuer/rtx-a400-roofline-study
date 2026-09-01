import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def print_header():
    print("=" * 80)
    print("   RTX A400 (SM86) FLASHATTENTION-2 (SDPA) vs EAGER ATTENTION STUDY   ")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)} (SM86 Ampere)")
    print("=" * 80 + "\n")

def run_sdpa_study():
    print_header()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Prompts desafiadores: 512, 1024 e 2048 TOKENS!
    prompt_lengths = [512, 1024, 2048]
    gen_tokens = 64
    base_text = "NVIDIA CUDA GPU architecture memory bandwidth Tensor Core FlashAttention SDPA " * 400
    
    attn_implementations = ["eager", "sdpa"]
    results = []
    
    for length in prompt_lengths:
        print(f"\n==================== PROMPT LENGTH: {length} TOKENS ====================")
        
        tokens = tokenizer(base_text, return_tensors="pt", max_length=length, truncation=True).to("cuda")
        actual_input_len = tokens.input_ids.shape[1]
        
        for attn_impl in attn_implementations:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            attn_label = "FlashAttention-2 (SDPA)" if attn_impl == "sdpa" else "Eager Math Attention"
            print(f"--> Testing [{attn_label}] at {actual_input_len} tokens...")
            
            try:
                # Carregar modelo com a atenção selecionada
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, 
                    torch_dtype=torch.bfloat16, 
                    attn_implementation=attn_impl,
                    device_map="cuda",
                    low_cpu_mem_usage=True
                )
                
                # Warmup
                with torch.inference_mode():
                    _ = model.generate(**tokens, max_new_tokens=2)
                torch.cuda.synchronize()
                
                # Medir TTFT (Prefill Latency)
                t_prefill_start = time.perf_counter()
                with torch.inference_mode():
                    _ = model(**tokens)
                torch.cuda.synchronize()
                ttft_ms = (time.perf_counter() - t_prefill_start) * 1000.0
                
                # Medir Full Generation
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
                
                decode_time_sec = total_time - (ttft_ms / 1000.0)
                tpot_ms = (decode_time_sec / actual_gen_len) * 1000.0
                decode_tok_s = actual_gen_len / decode_time_sec
                
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
                
                print(f"    [Result {attn_impl.upper()}]")
                print(f"    - TTFT (Prefill):       {ttft_ms:.2f} ms")
                print(f"    - TPOT (Decode):        {tpot_ms:.2f} ms/tok ({decode_tok_s:.1f} tok/s)")
                print(f"    - Peak VRAM Allocated:  {peak_vram_mb:.1f} MB")
                
                results.append({
                    "prompt_len": actual_input_len,
                    "attn": attn_impl,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "tok_s": decode_tok_s,
                    "peak_vram": peak_vram_mb,
                    "status": "OK"
                })
                
                del model
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    [Result {attn_impl.upper()}] FAILED / OOM: {e}")
                results.append({
                    "prompt_len": actual_input_len,
                    "attn": attn_impl,
                    "ttft_ms": 0,
                    "tpot_ms": 0,
                    "tok_s": 0,
                    "peak_vram": 0,
                    "status": "OOM / Failed"
                })
                gc.collect()
                torch.cuda.empty_cache()

    # Exibir Tabela Consolidada
    print("\n" + "=" * 80)
    print("            FLASHATTENTION-2 (SDPA) vs EAGER SUMMARY MATRIX             ")
    print("=" * 80)
    print(f"{'Prompt Len':<10} | {'Impl':<8} | {'TTFT (ms)':<12} | {'TPOT (ms)':<12} | {'Decode Tok/s':<12} | {'VRAM Peak':<10}")
    print("-" * 80)
    for r in results:
        if r['status'] == "OK":
            print(f"{r['prompt_len']:<10} | {r['attn'].upper():<8} | {r['ttft_ms']:>8.2f} ms | {r['tpot_ms']:>8.2f} ms | {r['tok_s']:>8.1f} tok/s | {r['peak_vram']:>6.1f} MB")
        else:
            print(f"{r['prompt_len']:<10} | {r['attn'].upper():<8} | OOM / FAILED")
    print("=" * 80)

if __name__ == "__main__":
    run_sdpa_study()