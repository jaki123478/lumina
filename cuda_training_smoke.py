"""Smoke test CUDA per Lumina: verifica forward/backward senza modificare checkpoint."""
import torch
from modeling_lumina_next import LuminaNextForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise SystemExit("CUDA non disponibile: nessun test di training eseguito")

model = LuminaNextForCausalLM.from_pretrained("checkpoints_lumina_next").to(device=device, dtype=torch.bfloat16)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
x = torch.randint(0, 32000, (1, 128), device=device)
y = torch.roll(x, shifts=-1, dims=1)
for step in range(1, 4):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print(f"step={step} loss={loss.item():.4f} vram_mb={torch.cuda.memory_allocated()/1024**2:.1f}", flush=True)
print("CUDA_TRAINING_SMOKE_OK")
