import time
import gc
import json
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
GEN_TOKENS = 8

def query_nvidia_smi():
    fields = [
        "memory.used",
        "memory.free",
        "memory.total",
        "utilization.gpu",
        "utilization.memory",
        "pcie.link.gen.current",
        "pcie.link.width.current",
    ]

    cmd = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]

    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        vals = [x.strip() for x in out.split(",")]
        return dict(zip(fields, vals))
    except Exception as e:
        return {"error": str(e)}

def torch_mem_snapshot():
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        free_mb = free_b / (1024 ** 2)
        total_mb = total_b / (1024 ** 2)
    except Exception:
        free_mb = None
        total_mb = None

    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "max_reserved_mb": torch.cuda.max_memory_reserved() / (1024 ** 2),
        "mem_get_free_mb": free_mb,
        "mem_get_total_mb": total_mb,
        "nvidia_smi": query_nvidia_smi(),
    }

def print_snapshot(label, snap):
    smi = snap["nvidia_smi"]

    print(f"    [{label}]")
    print(f"      torch allocated:      {snap['allocated_mb']:.1f} MB")
    print(f"      torch reserved:       {snap['reserved_mb']:.1f} MB")
    print(f"      torch max allocated:  {snap['max_allocated_mb']:.1f} MB")
    print(f"      torch max reserved:   {snap['max_reserved_mb']:.1f} MB")

    if snap["mem_get_free_mb"] is not None:
        print(f"      torch mem_get_info:   free={snap['mem_get_free_mb']:.1f} MB / total={snap['mem_get_total_mb']:.1f} MB")

    if "error" not in smi:
        print(
            f"      nvidia-smi memory:    used={smi.get('memory.used')} MB / "
            f"free={smi.get('memory.free')} MB / total={smi.get('memory.total')} MB"
        )
        print(
            f"      nvidia-smi util:      gpu={smi.get('utilization.gpu')}% / "
            f"mem={smi.get('utilization.memory')}%"
        )
        print(
            f"      PCIe link:            gen={smi.get('pcie.link.gen.current')} / "
            f"width=x{smi.get('pcie.link.width.current')}"
        )
    else:
        print(f"      nvidia-smi error:     {smi['error']}")

def main():
    print("=" * 90)
    print(" RTX A400 WDDM PAGING / OVERSUBSCRIPTION TELEMETRY VALIDATION ")
    print("=" * 90)

    device = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    physical_mb = props.total_memory / (1024 ** 2)

    print(f"Device:              {device}")
    print(f"Physical VRAM:       {physical_mb:.1f} MB")
    print(f"PyTorch:             {torch.__version__}")
    print(f"CUDA runtime:        {torch.version.cuda}")
    print("=" * 90 + "\n")

    gc.collect()
    torch.cuda.empty_cache()

    print("--> Loading model BF16 + SDPA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
        low_cpu_mem_usage=True,
    )

    base_after_load = torch_mem_snapshot()
    print_snapshot("after model load", base_after_load)
    print()

    context_lengths = [1024, 2048, 3072, 4096]
    base_text = "NVIDIA CUDA GPU memory allocation KV cache paging WDDM PCIe oversubscription " * 1000

    results = []

    for ctx_len in context_lengths:
        print("\n" + "=" * 90)
        print(f"Context length target: {ctx_len}")
        print("=" * 90)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        tokens = tokenizer(
            base_text,
            return_tensors="pt",
            max_length=ctx_len,
            truncation=True,
        ).to("cuda")

        actual_ctx = tokens.input_ids.shape[1]

        snap_before = torch_mem_snapshot()
        print_snapshot("before generate", snap_before)

        try:
            # Tiny warmup
            with torch.inference_mode():
                _ = model.generate(**tokens, max_new_tokens=1, do_sample=False)
            torch.cuda.synchronize()

            torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(
                    **tokens,
                    max_new_tokens=GEN_TOKENS,
                    min_new_tokens=GEN_TOKENS,
                    do_sample=False,
                )
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            elapsed = t1 - t0
            tok_s = GEN_TOKENS / elapsed

            snap_after = torch_mem_snapshot()
            print_snapshot("after generate", snap_after)

            peak_alloc = snap_after["max_allocated_mb"]
            peak_reserved = snap_after["max_reserved_mb"]

            oversub_alloc_mb = max(0.0, peak_alloc - physical_mb)
            oversub_reserved_mb = max(0.0, peak_reserved - physical_mb)

            print()
            print(f"    Result:")
            print(f"      generated tokens:      {GEN_TOKENS}")
            print(f"      elapsed:               {elapsed:.2f} s")
            print(f"      throughput:            {tok_s:.3f} tok/s")
            print(f"      peak allocated:        {peak_alloc:.1f} MB ({(peak_alloc / physical_mb) * 100:.1f}% of physical)")
            print(f"      peak reserved:         {peak_reserved:.1f} MB ({(peak_reserved / physical_mb) * 100:.1f}% of physical)")
            print(f"      allocated over limit:  {oversub_alloc_mb:.1f} MB")
            print(f"      reserved over limit:   {oversub_reserved_mb:.1f} MB")

            results.append({
                "context_len": actual_ctx,
                "status": "success",
                "elapsed_s": elapsed,
                "tok_s": tok_s,
                "physical_vram_mb": physical_mb,
                "peak_allocated_mb": peak_alloc,
                "peak_reserved_mb": peak_reserved,
                "oversub_allocated_mb": oversub_alloc_mb,
                "oversub_reserved_mb": oversub_reserved_mb,
                "before": snap_before,
                "after": snap_after,
            })

        except torch.cuda.OutOfMemoryError as e:
            print(f"    HARD CUDA OOM at context {actual_ctx}: {e}")
            results.append({
                "context_len": actual_ctx,
                "status": "oom",
                "physical_vram_mb": physical_mb,
            })
            break

        except Exception as e:
            print(f"    FAILED at context {actual_ctx}: {e}")
            results.append({
                "context_len": actual_ctx,
                "status": "failed",
                "error": str(e),
                "physical_vram_mb": physical_mb,
            })
            break

    with open("wddm_paging_telemetry_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print(" WDDM PAGING TELEMETRY SUMMARY ")
    print("=" * 90)
    print(f"{'Ctx':<8} | {'Status':<8} | {'Tok/s':<10} | {'Peak Alloc':<14} | {'Peak Reserved':<14} | {'Over Alloc':<12}")
    print("-" * 90)

    for r in results:
        if r["status"] == "success":
            print(
                f"{r['context_len']:<8} | "
                f"{r['status']:<8} | "
                f"{r['tok_s']:<10.3f} | "
                f"{r['peak_allocated_mb']:>8.1f} MB | "
                f"{r['peak_reserved_mb']:>8.1f} MB | "
                f"{r['oversub_allocated_mb']:>8.1f} MB"
            )
        else:
            print(f"{r['context_len']:<8} | {r['status']:<8}")

    print("=" * 90)
    print("Saved: wddm_paging_telemetry_results.json")

if __name__ == "__main__":
    main()