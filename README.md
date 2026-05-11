# ⚡ Viking Engine: Sub-Linear LoRA Scaling for Ostris AI-Toolkit

**Double-buffered async CUDA memory manager + bf16 precision forcing**  
**6.24 billion trainable parameters on a single RTX 4090. Zero OOM.**

*Orakul Studio — Chernihiv, Ukraine 🇺🇦*

[history in the making](https://github.com/OrakulStudio/AI-Toolkit-Memory-Manager-Fix)

---

> ## ⚠️ CRITICAL: READ BEFORE ANYTHING ELSE
>
> **This configuration WILL NOT WORK without the rewritten memory manager.**
>
> The speeds shown in this article are only achievable when
> `manager_modules.py` is replaced with the Viking Engine version.
> Without it, high-rank training will either OOM or run at baseline speeds
> (179+ sec/iter at rank 1024).
>
> **Required files:**
> - [manager_modules.py](https://github.com/OrakulStudio/ai-toolkit-Ostris-bonememory/blob/main/core/manager_modules.py) ← the engine (mandatory)
> - [BaseSDTrainProcess.py](https://github.com/OrakulStudio/ai-toolkit-Ostris-bonememory/blob/main/core/BaseSDTrainProcess.py) ← bf16 precision patch (mandatory)
>
> Installing one without the other will not produce these results.

---

## Works With All Models in ai-toolkit

Flux2 was chosen as the **test model** because it is the heaviest
available — 32 billion parameters — making it the most demanding
benchmark for any memory optimization.

**This engine works with every model supported by ostris/ai-toolkit:**

- ✅ FLUX.1 / FLUX.2
- ✅ Stable Diffusion 1.x / 2.x
- ✅ SDXL
- ✅ Stable Diffusion 3 / 3.5
- ✅ Video models (Wan, HunyuanVideo, etc.)
- ✅ Any future model added to ai-toolkit

Flux2 (32B) was used for benchmarks because if it works there,
it works everywhere.

---

## The Numbers

### Production Training — Rank 32 / Alpha 64 (recommended)

| Metric | Value |
|---|---|
| Speed | **5.92 – 6.57 sec/iter** |
| Full 1000-step training | **~2.5 hours** |
| Video-LoRA and lighter models | **~2 sec/iter** |

### Scaling Benchmark — Flux2-dev, RTX 4090

| Rank | Trainable Params | Speed | Theoretical Worst | Efficiency Gain |
|------|-----------------|-------|-------------------|-----------------|
| 16 | 97,517,568 | **5.92 s/it** | 5.92 s | 1× (baseline) |
| 512 | 3,120,562,176 | **~14 s/it** | ~94 s (bf16) | **6.7×** |
| 1024 | 6,241,124,352 | **~47 s/it** | ~188 s (bf16) | **4×+** |
| 1024 (stress) | 6,241,124,352 | **stable, 0 OOM** | OOM expected | **∞** |

> **Note:** Rank 1024 with bf16 forcing not yet fully benchmarked.
> Results pending. Rank 512 bf16 result (~14s) confirmed.

### The Sub-Linear Scaling Proof

```
Parameters ×32   →   Speed ×2.4 only   (rank 16 → rank 512, bf16)
Parameters ×64   →   Speed ×8 only     (rank 16 → rank 1024, bf32)
```

This is the key result. As rank increases, the engine becomes
**more efficient relative to parameter count** — not less.

**Why:** Higher rank = larger weight matrices = longer GPU compute per layer
= more time to hide CPU→GPU transfer latency via double-buffering.
The overlap efficiency increases with rank.

---

## Proof of Quality

Models trained with Viking Engine at rank 512 are published on CivitAI.
They demonstrate that high-rank training with this engine produces
superior quality — not just speed.

🔗 [Orakul Studio on CivitAI](https://civitai.com/user/orakul_storm)

If the speed results raise doubts — the trained models are the answer.

---

## Architecture: Two Engines in One File

### The Problem

Standard ai-toolkit layer offloading is sequential:

```
GPU computes layer N     ████████████████░░░░░░░░
Transfer weights N+1                     ████████
GPU computes layer N+1                           ████████████████
                         ↑ GPU idle here ↑
```

At rank 512: weight matrices = hundreds of MB per layer.
Sequential transfer at this scale = 37+ sec/iter without optimization.

### Engine 1: Direct Path (Small Ranks — 16, 32)

For small ranks, weight matrices are light. The overhead of CUDA Events
and double-buffering exceeds the transfer time itself. Direct path wins:

```python
# LinearLayerMemoryManager._f()
# Small weights → direct non-blocking transfer, zero event overhead
w = self.m.weight.to(device, non_blocking=True)
w = _dequant(w, dtype)
return F.linear(x, w, b)
```

### Engine 2: Double-Buffered Async (Large Ranks — 512, 1024)

For large ranks, compute time per layer is long enough to completely
hide the transfer latency. Double-buffering activates:

```python
# _BouncingLinearFn.forward()
# Two buffers — ping and pong
# Transfer stream runs PARALLEL to compute stream

with torch.cuda.stream(state["transfer_stream"]):
    state["transfer_stream"].wait_event(state["compute_forward_start_event"])
    w = weight_cpu.to(device, non_blocking=True)   # non-blocking DMA
    state["w_buffers"][idx] = _dequant(w, dtype)
    state["forward_clk"] ^= 1                       # 0→1→0→1 ping-pong
    state["transfer_forward_finished_event"].record()

torch.cuda.current_stream().wait_event(state["transfer_forward_finished_event"])
state["compute_forward_start_event"].record()
return F.linear(x, state["w_buffers"][idx], state["b_buffers"][idx])
```

**Result: transfer disappears from the profiler entirely.**

```
# Rank 32 iteration breakdown:
  backward:       3.85s  ← GPU computing gradients
  predict_unet:   2.01s  ← forward pass
  optimizer_step: 0.08s  ← weight update
  transfer:       0.00s  ← hidden inside compute ✓
```

### The Precision Patch (BaseSDTrainProcess.py)

One line. Added before `network.apply_to()`:

```python
# Викинг метод ранг 1024
# todo switch everything to proper mixed precision like this
self.network.force_to(self.device_torch, dtype=torch.bfloat16)
```

**What this does:**
- Forces all LoRA matrices from float32 → bfloat16
- Weight size: halved (4 bytes → 2 bytes per parameter)
- DMA transfer time: halved
- Overlap efficiency: further increased
- Numerical precision: sufficient for stable training (bf16 range)

**Impact at rank 512:**
- Before: ~37 sec/iter (float32 matrices)
- After: ~14 sec/iter (bfloat16 matrices)
- **2.7× speedup from one line**

The `# todo` comment is in the original ai-toolkit source.
It marks the direction the framework needed to go.
We went there.

### Pinned Memory — DMA Without CPU Cache

```python
def _ensure_cpu_pinned(t):
    if not t.is_pinned():
        t = t.pin_memory()
    return t
```

Standard RAM can be paged out by the OS at any time.
Pinned (page-locked) memory cannot.
GPU DMA controller reads it directly from DRAM — no CPU cache copy.
Another multiplier on transfer bandwidth.

### Smart Text Encoder Orchestration

```
1. Load Flux2 (32B) + Mistral-24B text encoder
2. Mistral encodes all training captions → cache to disk
3. ***** UNLOADING TEXT ENCODER *****
4. Mistral released → RAM freed
5. Training begins with Flux2 only + Viking Engine
```

Most users keep everything in memory simultaneously and hit OOM.
Sequential loading + unloading makes the impossible possible.

---

## Real Training Logs

### Rank 512 warmup (new — bf16 precision):
```
testrank512: step 51 →  71.20s/it  (cold start)
testrank512: step 53 →  32.60s/it  (pipeline filling)
testrank512: step 60 →  19.12s/it  (overlap activating)
testrank512: step 70 →  16.17s/it  (stabilizing)
testrank512: step 80 →  15.19s/it
testrank512: step 86 →  14.86s/it  (still dropping...)
```
[log_rank_512](https://github.com/OrakulStudio/ai-toolkit-Ostris-bonememory/blob/main/logs/log_rank_512.txt)

[original benchmark videos](https://github.com/OrakulStudio/ai-toolkit-Ostris-bonememory/releases/download/media-assets/Rank512.mp4)

### Rank 1024 warmup (bf32):
```
testrank1024: step 1  → 181.33s/it  (cold start)
testrank1024: step 5  →  69.01s/it  (pipeline filling)
testrank1024: step 10 →  54.80s/it
testrank1024: step 20 →  47.46s/it  (stabilizing)
testrank1024: step 22 →  47.01s/it  (stable)
```
[log_rank_1024](https://github.com/OrakulStudio/ai-toolkit-Ostris-bonememory/blob/main/logs/log_rank_1024.txt)

[log_rank_1024_аlfa64](https://github.com/OrakulStudio/ai-toolkit-Ostris-bonememory/blob/main/logs/log_rank_1024_%D0%B0lfa64.txt)

<img width="3839" height="2159" alt="Снимок экрана 2026-05-09 175606" src="https://github.com/user-attachments/assets/d4c05e71-41e1-4d17-a9ac-6dce69040302" />


### Total training parameters confirmed:
```
Rank 512:  Total training paramiters: 3,120,562,176
Rank 1024: Total training paramiters: 6,241,124,352
```

6.24 billion trainable parameters. Single RTX 4090. Zero crashes.
19.5% of the entire Flux2 model being trained simultaneously.

---

## Scalability to Server Hardware

This is not a consumer GPU workaround. It is architecture.

```python
# Consumer: CPU RAM → GPU VRAM (PCIe)
w = weight_cpu.to(device, non_blocking=True)

# Server NVLink: GPU_0 VRAM → GPU_1 VRAM
w = weight_gpu0.to(device_1, non_blocking=True)

# Tensor parallelism: shard assembly
w_shard = weight_shard.to(target_device, non_blocking=True)
```

Same double-buffering. Same CUDA Streams. Same Events.
The source changes. The architecture is identical.

On H100 NVLink clusters, this pattern eliminates inter-GPU transfer
stalls in tensor parallelism — the exact same way it eliminates
CPU-GPU transfer stalls on consumer hardware.

---

## Installation

### Step 1 — Clone ai-toolkit

```bash
git clone https://github.com/ostris/ai-toolkit
cd ai-toolkit
pip install -r requirements.txt
```

### Step 2 — Replace the memory manager

```bash
# Backup original
cp toolkit/manager_modules.py toolkit/manager_modules_original.py

# Replace with Viking Engine
cp path/to/viking/manager_modules.py toolkit/manager_modules.py
```

### Step 3 — Apply precision patch

In `jobs/process/BaseSDTrainProcess.py`, find the network initialization
block (around line 1774) and add one line:

```python
# After network is created, before apply_to():
# Викинг метод ранг 1024
# todo switch everything to proper mixed precision like this
self.network.force_to(self.device_torch, dtype=torch.bfloat16)
```

### Step 4 — Recommended config (rank 32, balanced)

```yaml
network:
  type: lora
  linear: 32
  linear_alpha: 64
  conv: 32
  conv_alpha: 64
  lokr_full_rank: true
  lokr_factor: -1
```

Expected result on RTX 4090 + Flux2: **5.92 – 6.57 sec/iter**

---

## Files

| File | Purpose | Required |
|---|---|---|
| `toolkit/manager_modules.py` | Viking double-buffer engine | ✅ Yes |
| `jobs/process/BaseSDTrainProcess.py` | bf16 precision patch | ✅ Yes |
| PyTorch CUDA patches | sm_89 support, stream priority, pin memory | Recommended |

PyTorch patches: `_device_limits.py`, `streams.py`, `_pin_memory_utils.py`  
→ Enable RTX 4090 (sm_89) native support + high-priority CUDA streams

---

## Context

ostris/ai-toolkit is the most widely used open-source LoRA training
framework. Thousands of people use it daily on consumer hardware.
The default memory management is sequential.

[ostris](https://github.com/ostris) requested this code for integration
into the main repository. The ticket is open.

This was built in Chernihiv, Ukraine — in a basement, under artillery
fire, on an RTX 4090. No datacenter. No cluster. No team.

The `# todo switch everything to proper mixed precision like this`
comment was already in the source. We read it, understood it,
and implemented it deeper than the original author expected.

**Architecture matters more than hardware.**

---

## What's Next

- [ ] Rank 1024 + bf16 combined benchmark (in progress)
- [ ] Conv2d layer support in double-buffer engine
- [ ] Multi-GPU weight streaming via NVLink / PCIe
- [ ] Adaptive rank detection (auto-select engine path)
- [ ] Upstream PR to ostris/ai-toolkit

---

## GitHub

[https://github.com/OrakulStudio](https://github.com/OrakulStudio)

---

*The smell of the iron is stable. The system is running. 🦊*

*Chernihiv, Ukraine 🇺🇦 · Orakul Studio · 2026*
