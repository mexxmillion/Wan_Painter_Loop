import gc
import torch
import comfy.model_management
from .nodes_loop_sampler import clear_lora_cache


class WanLoopClearLoraCache:
    """Clear the WanLoopSampler LoRA cache.

    The sampler caches loaded LoRA files and patched models in RAM across runs.
    Use this node if you replace a LoRA file on disk mid-session, or to free
    the CPU RAM used by cached patches.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "anything": ("*", {"tooltip": "Pass any connection through to trigger before downstream nodes"}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "clear"
    CATEGORY = "WanLoop"
    OUTPUT_NODE = True
    DESCRIPTION = "Flush the WanLoopSampler LoRA file + patch cache. Use if you updated a LoRA file on disk."

    def clear(self, anything=None):
        clear_lora_cache()
        print("  [WanLoop] LoRA cache cleared.")
        return (anything,)


class WanLoopPurgeVRAM:
    """Force purge all models from VRAM and clear cache.

    Use after OOM errors to recover without restarting ComfyUI.
    Wire any input through it to trigger before your workflow runs,
    or use it standalone.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "anything": ("*", {"tooltip": "Pass any connection through to trigger purge before downstream nodes"}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "purge"
    CATEGORY = "WanLoop"
    OUTPUT_NODE = True
    DESCRIPTION = "Force unload all models and clear VRAM cache. Use to recover from OOM without restarting ComfyUI."

    def purge(self, anything=None):
        device = comfy.model_management.get_torch_device()
        free_before = comfy.model_management.get_free_memory(device) / (1024**2)

        comfy.model_management.unload_all_models()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        comfy.model_management.soft_empty_cache()

        free_after = comfy.model_management.get_free_memory(device) / (1024**2)
        freed = free_after - free_before
        print(f"  [PurgeVRAM] Freed {freed:.0f}MB — VRAM: {free_before:.0f}MB → {free_after:.0f}MB")

        return (anything,)
