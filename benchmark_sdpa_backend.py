import time
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)

# Representative attention shape
B = 1
H = 12
D = 128

for N in [512, 1024, 2048]:
    print(f"\n=== Sequence length {N} ===")

    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    for name, backend in [
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("MATH", SDPBackend.MATH),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
    ]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            with sdpa_kernel(backend):
                # Warmup
                for _ in range(5):
                    out = F.scaled_dot_product_attention(q, k, v)

                torch.cuda.synchronize()

                iterations = 20
                start = time.perf_counter()

                for _ in range(iterations):
                    out = F.scaled_dot_product_attention(q, k, v)

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

            latency_ms = elapsed / iterations * 1000
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2

            print(
                f"{name:<22} "
                f"{latency_ms:8.3f} ms | "
                f"peak allocated {peak_mb:8.1f} MB"
            )

        except Exception as e:
            print(f"{name:<22} UNSUPPORTED: {e}")

    del q, k, v