import gc
import torch
import comfy.model_management


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
