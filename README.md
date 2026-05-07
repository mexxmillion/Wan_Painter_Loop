# ComfyUI-WanLoopEasy

Multi-segment looping video generation for **Wan 2.2 Image-to-Video** models in ComfyUI. Generate seamless long-form videos by chaining multiple ~5-second segments together, with per-segment prompts and LoRAs.

![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-blue)

## Features

- **Multi-segment looping** — Split prompts with `---` to generate sequential video segments that chain seamlessly (last frame becomes the next segment's start image)
- **Dual-model workflow** — HIGH-noise model handles initial denoising, LOW-noise model refines. Split point is configurable
- **Two I2V modes** — *Painter I2V* (motion-enhanced latents + reference conditioning) or *Regular I2V* (stock WanImageToVideo)
- **Per-segment LoRAs** — Apply base LoRAs to all segments plus unique LoRAs per segment (up to 6 segments)
- **Two-level LoRA cache** — File cache + patch cache eliminates redundant disk I/O and model patching across runs
- **Advanced guidance** — Optional NAG (Normalized Attention Guidance) and CFGZeroStar post-CFG correction
- **VRAM management** — Optional purge between passes/segments for 24GB GPUs; disable on 48GB+ for speed
- **Intermediate saves** — Optionally save each segment as a WEBP video as soon as it finishes
- **Dry-run mode** — Test your wiring and LoRA loading without waiting for actual sampling

## Installation

### ComfyUI Manager (Recommended)

Search for `ComfyUI-WanLoopEasy` in the ComfyUI Manager and install.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mexxmillion/Wan_Painter_Loop.git ComfyUI-WanLoopEasy
```

No additional pip dependencies required — uses only what ComfyUI already provides.

## Nodes

### Wan I2V Loop Sampler

The main node. Orchestrates multi-segment video generation.

| Input | Type | Description |
|-------|------|-------------|
| `model_high` | MODEL | HIGH-noise Wan 2.2 I2V model (apply SageAttention before connecting) |
| `model_low` | MODEL | LOW-noise Wan 2.2 I2V model (apply SageAttention before connecting) |
| `clip` | CLIP | UMT5-XXL text encoder |
| `vae` | VAE | Wan 2.1 VAE |
| `start_image` | IMAGE | First frame / reference image |
| `prompt` | STRING | Use `---` to split into segments. Each segment = one ~5s generation |
| `negative_prompt` | STRING | Negative prompt (shared across all segments) |
| `width` / `height` | INT | Output resolution (default: 832x480) |
| `length` | INT | Frames per segment (81 = ~5s at 16fps) |
| `seed` | INT | Random seed |
| `total_steps` | INT | Total sampling steps (default: 7) |
| `split_step` | INT | HIGH runs steps 0→split-1, LOW runs split→total-1 (default: 3) |
| `cfg` | FLOAT | CFG scale (default: 1.0) |
| `shift` | FLOAT | Flow shift parameter (default: 5.0) |
| `motion_amplitude` | FLOAT | Motion enhancement scale, Painter mode only (1.0–2.0, default: 1.5) |
| `resize_mode` | ENUM | How to fit start image: stretch, crop, pad, or resize |
| `sampler_name` | ENUM | Sampling algorithm (default: euler) |
| `scheduler` | ENUM | Noise schedule (default: simple) |
| `dry_run` | BOOL | Skip sampling, return dummy frames for testing |
| `save_intermediates` | BOOL | Save each segment as WEBP to output folder |
| `purge_vram` | BOOL | Purge VRAM between passes (recommended for ≤24GB GPUs) |

**Optional inputs:**

| Input | Description |
|-------|-------------|
| `i2v_mode` | "Painter I2V" (enhanced motion) or "Regular I2V" (stock conditioning) |
| `nag_enable` | Enable Normalized Attention Guidance |
| `nag_scale` | NAG strength (default: 11.0) |
| `nag_alpha` | NAG mixing balance (default: 0.25) |
| `nag_tau` | NAG clipping threshold (default: 2.5) |
| `cfg_zero_star` | Enable CFGZeroStar correction (pairs well with NAG) |
| `lora_stack` | Base LoRAs applied to ALL segments |
| `lora_stack_1`–`6` | Per-segment LoRAs (on top of base) |

**Outputs:** `images` (all frames concatenated), `last_frame`, `frame_count`

### Wan Loop LoRA Stack

Builds HIGH/LOW LoRA pairs for the dual-model workflow.

- 4 chainable LoRA slots per node
- Each slot has separate HIGH and LOW LoRA selections
- Shared strength control per slot
- Stackable — chain multiple nodes together via the `lora_stack` input

### Wan Loop Resolution

Quick resolution preset selector.

Presets: 832x480 (Landscape), 480x832 (Portrait), 640x640 (Square), 576x1024 (Tall), 1024x576 (Wide), Custom

### Purge VRAM

Force-unloads all models and clears GPU/CPU caches. Use for OOM recovery without restarting ComfyUI.

### Clear LoRA Cache

Flushes the sampler's persistent LoRA caches. Use after replacing LoRA files on disk mid-session.

## Recommended Models

| Model | Purpose | Notes |
|-------|---------|-------|
| Wan 2.2 I2V 480p (14B) | HIGH-noise model | Use with SageAttention for speed |
| Wan 2.2 I2V 480p (14B) | LOW-noise model | Same checkpoint, different LoRAs |
| UMT5-XXL | Text encoder | Standard Wan CLIP |
| wan_2.1_vae | VAE | Shared across Wan versions |

Load both models with `fp8_e4m3fn` weight type for 24GB GPUs. Apply SageAttention to both before connecting to the sampler.

## Quick Start

1. Load the example workflow: `Wan_I2V_Loop_Easy.json` (included in this package)
2. Load your Wan 2.2 I2V model twice (HIGH + LOW)
3. Apply SageAttention to both model outputs
4. Connect CLIP (umt5_xxl) and VAE (wan_2.1_vae)
5. Load a start image
6. Write your prompt with `---` separating segments:
   ```
   A woman walks through a garden, flowers blooming around her
   ---
   She reaches a fountain, water sparkling in sunlight
   ---
   She sits on a bench and reads a book peacefully
   ```
7. Queue and wait — each segment takes ~60-90s on an RTX 3090

## Tips

- **VRAM (24GB):** Keep `purge_vram` enabled. Use `fp8_e4m3fn` weight type. Close other GPU apps.
- **VRAM (48GB+):** Disable `purge_vram` and use `--highvram` flag for faster generation (models stay resident).
- **Motion:** `motion_amplitude` of 1.5 gives natural movement. Go up to 2.0 for more dynamic scenes. Only applies in Painter I2V mode.
- **Quality:** Enable both `nag_enable` and `cfg_zero_star` for best prompt adherence. NAG scale of 11 is a good default.
- **LoRAs:** Use the LoRA Stack node to apply style LoRAs globally, and per-segment LoRA inputs for character/scene changes.
- **Dry run:** Test complex workflows with `dry_run=True` first — validates all connections and LoRA loading in seconds.
- **Segments:** Each segment at 81 frames = ~5 seconds at 16fps. Chain as many as you need.

## Example Workflow

An example workflow is included as `Wan_I2V_Loop_Easy.json`. Drag and drop it into ComfyUI to load.

## License

MIT
