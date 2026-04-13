import gc
import math
import re
import torch
import torch.nn.functional as F
import logging

import comfy.model_management
import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import latent_preview
import node_helpers

logger = logging.getLogger("WanLoopSampler")

RESIZE_MODES = ["stretch", "crop (center)", "crop (top)", "crop (bottom)", "pad (black)", "pad (edge)", "resize (fit)"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_image(image, width, height, mode="stretch", upscale_method="bilinear"):
    """Resize image tensor [B, H, W, C] to target width/height with various modes.

    Modes:
        stretch       - Direct resize, ignores aspect ratio
        crop (center) - Scale to fill, crop excess from center
        crop (top)    - Scale to fill, crop from top
        crop (bottom) - Scale to fill, crop from bottom
        pad (black)   - Scale to fit, pad with black
        pad (edge)    - Scale to fit, pad by repeating edge pixels
        resize (fit)  - Scale to fit within dimensions, no pad (output may be smaller)
    """
    # Ensure 4D [B, H, W, C]
    while image.ndim > 4:
        image = image.squeeze(0)
    if image.ndim == 3:
        image = image.unsqueeze(0)

    B, H, W, C = image.shape

    if H == height and W == width:
        return image

    # Convert to [B, C, H, W] for processing
    img = image.movedim(-1, 1)

    if mode == "stretch":
        img = comfy.utils.common_upscale(img, width, height, upscale_method, "disabled")

    elif mode.startswith("crop"):
        # Scale so the smaller dimension fills the target, then crop
        scale = max(width / W, height / H)
        new_w = round(W * scale)
        new_h = round(H * scale)
        img = comfy.utils.common_upscale(img, new_w, new_h, upscale_method, "disabled")

        # Crop to exact target
        if "top" in mode:
            img = img[:, :, :height, (new_w - width) // 2 : (new_w - width) // 2 + width]
        elif "bottom" in mode:
            img = img[:, :, new_h - height:, (new_w - width) // 2 : (new_w - width) // 2 + width]
        else:  # center
            y_off = (new_h - height) // 2
            x_off = (new_w - width) // 2
            img = img[:, :, y_off:y_off + height, x_off:x_off + width]

    elif mode.startswith("pad") or mode.startswith("resize"):
        # Scale so the larger dimension fits, then pad (or just return for resize fit)
        scale = min(width / W, height / H)
        new_w = max(1, round(W * scale))
        new_h = max(1, round(H * scale))

        # Ensure divisible by 2 for clean padding
        new_w = new_w - (new_w % 2)
        new_h = new_h - (new_h % 2)
        if new_w == 0:
            new_w = 2
        if new_h == 0:
            new_h = 2

        img = comfy.utils.common_upscale(img, new_w, new_h, upscale_method, "disabled")

        if mode.startswith("resize"):
            # Just fit, no padding — output exact scaled size
            pass
        else:
            # Pad to exact target size
            pad_left = (width - new_w) // 2
            pad_right = width - new_w - pad_left
            pad_top = (height - new_h) // 2
            pad_bottom = height - new_h - pad_top

            if "edge" in mode:
                img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
            else:
                # Black padding
                img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)

    else:
        # Fallback to stretch
        img = comfy.utils.common_upscale(img, width, height, upscale_method, "disabled")

    return img.movedim(1, -1)


def _apply_shift(model, shift):
    """Apply ModelSamplingSD3 shift to a model clone."""
    m = model.clone()

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
        pass

    model_sampling = ModelSamplingAdvanced(m.model.model_config)
    model_sampling.set_parameters(shift=shift)
    m.add_object_patch("model_sampling", model_sampling)
    return m


def _apply_loras(model, clip, lora_stack):
    """Apply LoRA configs [(name, strength_model, strength_clip), ...] to a single model."""
    if not lora_stack:
        return model, clip

    for lora_name, strength_model, strength_clip in lora_stack:
        if strength_model == 0 and strength_clip == 0:
            continue
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model, clip = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

    return model, clip


def _apply_loras_dual(model_high, model_low, clip, lora_stack):
    """Apply HIGH/LOW LoRA pairs to their respective models.

    lora_stack format: [(high_name, low_name, strength_model, strength_clip), ...]
    Either high or low can be None to skip that model.
    """
    if not lora_stack:
        return model_high, model_low, clip

    for high_name, low_name, strength_model, strength_clip in lora_stack:
        if strength_model == 0 and strength_clip == 0:
            continue

        if high_name is not None:
            lora_path = folder_paths.get_full_path_or_raise("loras", high_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_high, clip = comfy.sd.load_lora_for_models(model_high, clip, lora, strength_model, strength_clip)

        if low_name is not None:
            lora_path = folder_paths.get_full_path_or_raise("loras", low_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_low, clip = comfy.sd.load_lora_for_models(model_low, clip, lora, strength_model, strength_clip)

    return model_high, model_low, clip


def _encode_text(clip, text):
    """Encode text with CLIP, returns conditioning list."""
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens_scheduled(tokens)
    return output


def _create_i2v_conditioning(positive, negative, vae, start_image, width, height, length,
                              batch_size, motion_amplitude, resize_mode="crop (center)",
                              color_protect=True, correct_strength=0.01):
    """Port of PainterI2VAdvanced conditioning logic.

    Returns: (high_positive, high_negative, low_positive, low_negative, latent_dict)
    """
    latent = torch.zeros(
        [batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
        device=comfy.model_management.intermediate_device(),
    )

    positive_original = positive
    negative_original = negative

    if start_image is not None:
        # Ensure single frame and correct resolution
        start_image = start_image[:1]
        if start_image.shape[1] != height or start_image.shape[2] != width:
            start_image = _resize_image(start_image, width, height, mode=resize_mode)

        image = (
            torch.ones(
                (length, height, width, start_image.shape[-1]),
                device=start_image.device,
                dtype=start_image.dtype,
            )
            * 0.5
        )
        image[0] = start_image[0]

        concat_latent_image = vae.encode(image[:, :, :, :3])

        mask = torch.ones(
            (1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]),
            device=start_image.device,
            dtype=start_image.dtype,
        )
        mask[:, :, 0] = 0.0

        concat_latent_image_original = concat_latent_image.clone()

        if motion_amplitude > 1.0:
            base_latent = concat_latent_image[:, :, 0:1]
            gray_latent = concat_latent_image[:, :, 1:]

            diff = gray_latent - base_latent
            diff_mean = diff.mean(dim=(1, 3, 4), keepdim=True)
            diff_centered = diff - diff_mean

            scaled_latent = base_latent + diff_centered * motion_amplitude + diff_mean
            # NaN guard: if any values went haywire, fall back to original
            if torch.isnan(scaled_latent).any() or torch.isinf(scaled_latent).any():
                logger.warning("Motion amplitude produced NaN/Inf, falling back to unscaled latent")
                scaled_latent = gray_latent
            scaled_latent = torch.clamp(scaled_latent, -6, 6)
            concat_latent_image = torch.cat([base_latent, scaled_latent], dim=2)

            post_enhanced = concat_latent_image.clone()

            if color_protect and correct_strength > 0:
                orig_mean = concat_latent_image_original.mean(dim=(2, 3, 4))
                enhanced_mean = post_enhanced.mean(dim=(2, 3, 4))

                # Safe division to avoid NaN
                abs_orig = torch.abs(orig_mean)
                safe_denom = torch.where(abs_orig > 1e-4, abs_orig, torch.ones_like(abs_orig))
                mean_drift = torch.abs(enhanced_mean - orig_mean) / safe_denom
                problem_channels = mean_drift > 0.18

                if problem_channels.any():
                    drift_amount = enhanced_mean - orig_mean
                    correction = drift_amount * problem_channels.float() * correct_strength * 0.03

                    for b in range(batch_size):
                        for c in range(16):
                            if correction[b, c].abs() > 0:
                                post_enhanced[b, c] = torch.where(
                                    post_enhanced[b, c] > 0,
                                    post_enhanced[b, c] - correction[b, c],
                                    post_enhanced[b, c],
                                )

                orig_brightness = concat_latent_image_original.mean()
                enhanced_brightness = post_enhanced.mean()

                if enhanced_brightness < orig_brightness * 0.92 and enhanced_brightness.abs() > 1e-6:
                    brightness_boost = min((orig_brightness / enhanced_brightness).item(), 1.05)
                    if not (torch.isnan(torch.tensor(brightness_boost)) or torch.isinf(torch.tensor(brightness_boost))):
                        post_enhanced = torch.where(
                            post_enhanced < 0.5,
                            post_enhanced * brightness_boost,
                            post_enhanced,
                        )

                # Final NaN guard
                if torch.isnan(post_enhanced).any():
                    logger.warning("Color protect produced NaN, falling back to pre-correction latent")
                    post_enhanced = concat_latent_image.clone()

                concat_latent_image = torch.clamp(post_enhanced, -6, 6)

        # HIGH conditioning (motion-enhanced)
        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask}
        )

        # LOW conditioning (original, unenhanced)
        positive_original = node_helpers.conditioning_set_values(
            positive_original, {"concat_latent_image": concat_latent_image_original, "concat_mask": mask}
        )
        negative_original = node_helpers.conditioning_set_values(
            negative_original, {"concat_latent_image": concat_latent_image_original, "concat_mask": mask}
        )

        # Reference latents
        ref_latent = vae.encode(start_image[:, :, :, :3])
        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents": [ref_latent]}, append=True
        )
        negative = node_helpers.conditioning_set_values(
            negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True
        )
        positive_original = node_helpers.conditioning_set_values(
            positive_original, {"reference_latents": [ref_latent]}, append=True
        )
        negative_original = node_helpers.conditioning_set_values(
            negative_original, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True
        )

    out_latent = {"samples": latent}
    return positive, negative, positive_original, negative_original, out_latent


def _run_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                  latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None,
                  force_full_denoise=False):
    """Run a single KSampler pass. Mirrors common_ksampler from nodes.py."""
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler,
        positive, negative, latent_image,
        denoise=denoise, disable_noise=disable_noise,
        start_step=start_step, last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask, callback=callback,
        disable_pbar=disable_pbar, seed=seed,
    )
    out = latent.copy()
    out["samples"] = samples
    return out


# ---------------------------------------------------------------------------
# Main Node
# ---------------------------------------------------------------------------

class WanLoopSampler:
    """Wan 2.2 I2V Loop Sampler.

    Type one prompt, or use ``---`` to split into multiple loop segments:
        ``A woman walks forward --- She turns around --- She waves goodbye``
    Each segment generates ~5 seconds of video, chaining the last frame forward.

    Steps work like KSamplerAdvanced: ``total_steps`` is the full schedule,
    ``split_step`` is where HIGH-noise stops and LOW-noise takes over.
    E.g. total_steps=7, split_step=3 → HIGH runs steps 0-2, LOW runs steps 3-6.
    """

    MAX_SEGMENTS = 5

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high": ("MODEL", {"tooltip": "HIGH-noise Wan 2.2 I2V model (apply SageAttention before connecting)"}),
                "model_low": ("MODEL", {"tooltip": "LOW-noise Wan 2.2 I2V model (apply SageAttention before connecting)"}),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "start_image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Use --- to split into loop segments. Each segment = one ~5s video generation.",
                }),
                "negative_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "width": ("INT", {"default": 832, "min": 128, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 480, "min": 128, "max": 2048, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 257, "step": 4,
                                   "tooltip": "Frames per segment (81 = ~5s at 16fps)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "total_steps": ("INT", {"default": 7, "min": 2, "max": 100,
                                        "tooltip": "Total sampling steps for the full schedule"}),
                "split_step": ("INT", {"default": 3, "min": 1, "max": 99,
                                       "tooltip": "HIGH-noise runs steps 0 to split_step-1, LOW-noise runs split_step to total_steps-1"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01,
                                    "tooltip": "ModelSamplingSD3 shift value"}),
                "motion_amplitude": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 2.0, "step": 0.05}),
                "resize_mode": (RESIZE_MODES, {"default": "crop (center)",
                                               "tooltip": "How to fit the start image to target resolution"}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "dry_run": ("BOOLEAN", {"default": False,
                                        "tooltip": "Test mode: skips sampling, returns dummy frames. Use to verify wiring and LoRA loading without waiting."}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", {"tooltip": "Base LoRAs applied to ALL segments"}),
                "lora_stack_1": ("LORA_STACK", {"tooltip": "Extra LoRAs for segment 1 (on top of base)"}),
                "lora_stack_2": ("LORA_STACK", {"tooltip": "Extra LoRAs for segment 2 (on top of base)"}),
                "lora_stack_3": ("LORA_STACK", {"tooltip": "Extra LoRAs for segment 3 (on top of base)"}),
                "lora_stack_4": ("LORA_STACK", {"tooltip": "Extra LoRAs for segment 4 (on top of base)"}),
                "lora_stack_5": ("LORA_STACK", {"tooltip": "Extra LoRAs for segment 5 (on top of base)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("images", "last_frame", "frame_count")
    FUNCTION = "run_loop"
    CATEGORY = "WanLoop"
    DESCRIPTION = (
        "Wan 2.2 I2V multi-segment loop sampler. "
        "Use --- in the prompt to split into segments. "
        "Each segment chains the last frame into the next, "
        "running a HIGH-noise then LOW-noise pass."
    )

    def run_loop(self, model_high, model_low, clip, vae, start_image,
                 prompt, negative_prompt, width, height, length, seed,
                 total_steps, split_step, cfg, shift, motion_amplitude,
                 resize_mode, sampler_name, scheduler, dry_run=False,
                 lora_stack=None,
                 lora_stack_1=None, lora_stack_2=None, lora_stack_3=None,
                 lora_stack_4=None, lora_stack_5=None):

        # --- Validate step split ---
        if split_step >= total_steps:
            raise ValueError(f"split_step ({split_step}) must be less than total_steps ({total_steps})")

        # --- Parse prompt segments (split on 2+ dashes on their own line) ---
        segments = [s.strip() for s in re.split(r'\n\s*-{2,}\s*\n', prompt) if s.strip()]
        if not segments:
            raise ValueError("Prompt is empty. Please enter at least one prompt segment.")
        if len(segments) > self.MAX_SEGMENTS:
            logger.warning(f"Prompt has {len(segments)} segments, capping at {self.MAX_SEGMENTS}")
            segments = segments[: self.MAX_SEGMENTS]

        # --- Log the execution plan ---
        print(f"\n{'='*60}")
        print(f"WanLoopSampler: {len(segments)} segment(s) detected")
        print(f"  total_steps={total_steps}, split_step={split_step}")
        print(f"  resolution={width}x{height}, length={length}, cfg={cfg}")
        print(f"  shift={shift}, motion_amplitude={motion_amplitude}")
        print(f"  sampler={sampler_name}, scheduler={scheduler}")
        for j, seg in enumerate(segments):
            print(f"  Segment {j+1}: \"{seg[:100]}{'...' if len(seg)>100 else ''}\"")
        print(f"{'='*60}\n")

        per_iter_stacks = [lora_stack_1, lora_stack_2, lora_stack_3, lora_stack_4, lora_stack_5]

        # --- Purge VRAM before starting — unload everything from previous runs ---
        device = comfy.model_management.get_torch_device()
        print(f"  Purging VRAM before start... (free: {comfy.model_management.get_free_memory(device) / (1024**2):.0f}MB)")
        comfy.model_management.unload_all_models()
        gc.collect()
        comfy.model_management.soft_empty_cache()
        print(f"  VRAM after purge: {comfy.model_management.get_free_memory(device) / (1024**2):.0f}MB")

        # --- Take first frame only and resize ---
        current_image = _resize_image(start_image[:1], width, height, mode=resize_mode)

        # --- Pre-split the base lora_stack into HIGH-only and LOW-only lists ---
        high_base_loras = []
        low_base_loras = []
        if lora_stack:
            for high_name, low_name, strength_model, strength_clip in lora_stack:
                if high_name is not None:
                    high_base_loras.append((high_name, strength_model, strength_clip))
                if low_name is not None:
                    low_base_loras.append((low_name, strength_model, strength_clip))

        # --- Encode prompts with CLIP (before LoRAs, CLIP LoRA effect is minimal for Wan) ---
        neg_cond = _encode_text(clip, negative_prompt)

        all_frames = []

        for i, segment_prompt in enumerate(segments):
            logger.info(f"=== WanLoopSampler: Segment {i + 1}/{len(segments)} ===")
            logger.info(f"    Prompt: {segment_prompt[:80]}{'...' if len(segment_prompt) > 80 else ''}")

            # --- Pre-split per-segment LoRAs ---
            iter_stack = per_iter_stacks[i] if i < len(per_iter_stacks) else None
            high_iter_loras = list(high_base_loras)
            low_iter_loras = list(low_base_loras)
            if iter_stack:
                for high_name, low_name, strength_model, strength_clip in iter_stack:
                    if high_name is not None:
                        high_iter_loras.append((high_name, strength_model, strength_clip))
                    if low_name is not None:
                        low_iter_loras.append((low_name, strength_model, strength_clip))

            # --- Encode this segment's prompt ---
            pos_cond = _encode_text(clip, segment_prompt)

            # --- Create I2V conditioning ---
            high_pos, high_neg, low_pos, low_neg, latent = _create_i2v_conditioning(
                pos_cond, neg_cond, vae, current_image,
                width, height, length, batch_size=1,
                motion_amplitude=motion_amplitude,
                resize_mode=resize_mode,
            )

            if dry_run:
                # --- DRY RUN: skip sampling, create dummy frames ---
                logger.info(f"    DRY RUN: skipping sampling, generating {length} dummy frames")
                images = current_image.repeat(length, 1, 1, 1)
            else:
                iter_seed = seed + i
                device = comfy.model_management.get_torch_device()

                # --- HIGH-noise pass: load LoRAs, sample, then free ---
                high_lora_names = [name for name, _, _ in high_iter_loras] if high_iter_loras else []
                print(f"  [{i+1}] HIGH pass: steps 0→{split_step}, seed={iter_seed}, LoRAs={high_lora_names}")
                m_high = model_high.clone()
                if high_iter_loras:
                    m_high, _ = _apply_loras(m_high, clip, high_iter_loras)
                m_high = _apply_shift(m_high, shift)

                latent = _run_ksampler(
                    m_high, iter_seed, total_steps, cfg, sampler_name, scheduler,
                    high_pos, high_neg, latent,
                    start_step=0, last_step=split_step,
                )

                # Free HIGH model before loading LOW
                del m_high
                gc.collect()
                comfy.model_management.cleanup_models()
                comfy.model_management.soft_empty_cache()
                comfy.model_management.free_memory(
                    comfy.model_management.minimum_inference_memory(), device
                )
                print(f"  [{i+1}] HIGH done, freed. VRAM free: {comfy.model_management.get_free_memory(device) / (1024**2):.0f}MB")

                # --- LOW-noise pass: load LoRAs, sample, then free ---
                low_lora_names = [name for name, _, _ in low_iter_loras] if low_iter_loras else []
                print(f"  [{i+1}] LOW pass: steps {split_step}→{total_steps}, seed={iter_seed}, LoRAs={low_lora_names}")
                m_low = model_low.clone()
                if low_iter_loras:
                    m_low, _ = _apply_loras(m_low, clip, low_iter_loras)
                m_low = _apply_shift(m_low, shift)

                latent = _run_ksampler(
                    m_low, iter_seed, total_steps, cfg, sampler_name, scheduler,
                    low_pos, low_neg, latent,
                    disable_noise=True,
                    start_step=split_step, last_step=total_steps,
                    force_full_denoise=True,
                )

                del m_low

                # --- VAE decode ---
                images = vae.decode(latent["samples"])

                # Ensure 4D [F, H, W, C] — Wan VAE may return 5D
                while images.ndim > 4:
                    images = images.squeeze(0)

            # --- Extract last frame for next iteration (keep on CPU to save VRAM) ---
            current_image = images[-1:].cpu()

            # --- Collect frames (skip first on subsequent iterations to avoid duplicate) ---
            # Move to CPU immediately to free GPU VRAM for next segment
            if i > 0:
                all_frames.append(images[1:].cpu())
            else:
                all_frames.append(images.cpu())
            del images

            # --- Free memory between iterations (mimic ComfyUI's between-node cleanup) ---
            del high_pos, high_neg, low_pos, low_neg
            if not dry_run:
                del latent

            gc.collect()
            comfy.model_management.cleanup_models()
            comfy.model_management.soft_empty_cache()
            device = comfy.model_management.get_torch_device()
            comfy.model_management.free_memory(
                comfy.model_management.minimum_inference_memory(),
                device,
            )
            logger.info(f"    Segment {i + 1} done, VRAM free: {comfy.model_management.get_free_memory(device) / (1024**2):.0f}MB")

        # --- Concatenate all frames ---
        all_images = torch.cat(all_frames, dim=0)
        frame_count = all_images.shape[0]

        logger.info(f"=== WanLoopSampler complete: {frame_count} total frames from {len(segments)} segments ===")

        return (all_images, current_image, frame_count)
