"""
@title: ComfyUI-WanLoopEasy
@description: Simplified Wan 2.2 I2V looping workflow — multi-segment video with per-iteration LoRAs
@version: 1.0.0
"""

from .nodes_loop_sampler import WanLoopSampler
from .nodes_lora_stack import WanLoopLoraStack
from .nodes_resolution import WanLoopResolution

NODE_CLASS_MAPPINGS = {
    "WanLoopSampler": WanLoopSampler,
    "WanLoopLoraStack": WanLoopLoraStack,
    "WanLoopResolution": WanLoopResolution,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanLoopSampler": "Wan I2V Loop Sampler",
    "WanLoopLoraStack": "Wan Loop LoRA Stack",
    "WanLoopResolution": "Wan Loop Resolution",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
