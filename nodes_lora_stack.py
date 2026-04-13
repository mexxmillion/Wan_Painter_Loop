import folder_paths


class WanLoopLoraStack:
    """Builds a HIGH/LOW LoRA pair stack for Wan 2.2 I2V dual-model workflow.

    Each slot has a HIGH-noise LoRA and a LOW-noise LoRA with shared strength.
    Chainable — connect output to another stack's input to combine.
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "high_lora_1": (lora_list, {"default": "None", "tooltip": "LoRA for HIGH-noise model"}),
                "low_lora_1": (lora_list, {"default": "None", "tooltip": "LoRA for LOW-noise model"}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "high_lora_2": (lora_list, {"default": "None"}),
                "low_lora_2": (lora_list, {"default": "None"}),
                "strength_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "high_lora_3": (lora_list, {"default": "None"}),
                "low_lora_3": (lora_list, {"default": "None"}),
                "strength_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "high_lora_4": (lora_list, {"default": "None"}),
                "low_lora_4": (lora_list, {"default": "None"}),
                "strength_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK",),
            },
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "build"
    CATEGORY = "WanLoop"

    def build(self, high_lora_1, low_lora_1, strength_1,
              high_lora_2, low_lora_2, strength_2,
              high_lora_3, low_lora_3, strength_3,
              high_lora_4, low_lora_4, strength_4,
              lora_stack=None):
        stack = list(lora_stack) if lora_stack else []

        for high_name, low_name, strength in [
            (high_lora_1, low_lora_1, strength_1),
            (high_lora_2, low_lora_2, strength_2),
            (high_lora_3, low_lora_3, strength_3),
            (high_lora_4, low_lora_4, strength_4),
        ]:
            if strength == 0:
                continue
            if high_name != "None" or low_name != "None":
                stack.append((
                    high_name if high_name != "None" else None,
                    low_name if low_name != "None" else None,
                    strength,
                    strength,
                ))

        return (stack,)
