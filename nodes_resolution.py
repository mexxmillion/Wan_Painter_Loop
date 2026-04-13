class WanLoopResolution:
    """Resolution preset selector for Wan 2.2 I2V video generation."""

    PRESETS = {
        "832x480 (Landscape)": (832, 480),
        "480x832 (Portrait)": (480, 832),
        "640x640 (Square)": (640, 640),
        "576x1024 (Tall)": (576, 1024),
        "1024x576 (Wide)": (1024, 576),
        "Custom": None,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()), {"default": "832x480 (Landscape)"}),
                "custom_width": ("INT", {"default": 832, "min": 128, "max": 2048, "step": 16}),
                "custom_height": ("INT", {"default": 480, "min": 128, "max": 2048, "step": 16}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "resolve"
    CATEGORY = "WanLoop"

    def resolve(self, preset, custom_width, custom_height):
        dims = self.PRESETS.get(preset)
        if dims is None:
            return (custom_width, custom_height)
        return dims
