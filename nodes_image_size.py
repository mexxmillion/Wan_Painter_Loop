class WanLoopImageSize:
    """Calculate width and height from an input image's aspect ratio,
    using the given pixel value as the longest side."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "longest_side": ("INT", {"default": 832, "min": 128, "max": 2048, "step": 16}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calculate"
    CATEGORY = "WanLoop"

    def calculate(self, image, longest_side):
        src_h = image.shape[1]
        src_w = image.shape[2]

        if src_w >= src_h:
            w = longest_side
            h = round(longest_side * src_h / src_w / 16) * 16
        else:
            h = longest_side
            w = round(longest_side * src_w / src_h / 16) * 16

        return (max(w, 128), max(h, 128))
