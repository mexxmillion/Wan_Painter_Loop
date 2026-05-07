"""
@title: ComfyUI-WanLoopEasy
@description: Simplified Wan 2.2 I2V looping workflow — multi-segment video with per-iteration LoRAs
@version: 1.0.0
"""

from .nodes_loop_sampler import WanLoopSampler
from .nodes_lora_stack import WanLoopLoraStack
from .nodes_resolution import WanLoopResolution
from .nodes_purge_vram import WanLoopPurgeVRAM, WanLoopClearLoraCache
from .nodes_image_size import WanLoopImageSize

NODE_CLASS_MAPPINGS = {
    "WanLoopSampler": WanLoopSampler,
    "WanLoopLoraStack": WanLoopLoraStack,
    "WanLoopResolution": WanLoopResolution,
    "WanLoopPurgeVRAM": WanLoopPurgeVRAM,
    "WanLoopClearLoraCache": WanLoopClearLoraCache,
    "WanLoopImageSize": WanLoopImageSize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanLoopSampler": "Wan I2V Loop Sampler",
    "WanLoopLoraStack": "Wan Loop LoRA Stack",
    "WanLoopResolution": "Wan Loop Resolution",
    "WanLoopPurgeVRAM": "Purge VRAM",
    "WanLoopClearLoraCache": "Clear LoRA Cache",
    "WanLoopImageSize": "Wan Loop Image Size",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
