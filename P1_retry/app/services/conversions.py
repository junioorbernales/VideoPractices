class Image:
    def __init__(self, r: int, g: int, b: int):
        self.r = r
        self.g = g
        self.b = b
        self.y = 0
        self.u = 0
        self.v = 0

    def rgb_to_yuv(self):
        self.y = (0.257 * self.r) + (0.504 * self.g) + (0.098 * self.b) + 16
        self.u = (-0.148 * self.r) - (0.291 * self.g) + (0.439 * self.b) + 128
        self.v = (0.439 * self.r) - (0.368 * self.g) - (0.071 * self.b) + 128
        return {"Y": self.y, "U": self.u, "V": self.v}

    def yuv_to_rgb(self):
        self.b = 1.164 * (self.y - 16) + 2.018 * (self.u - 128)
        self.g = 1.164 * (self.y - 16) - 0.813 * (self.v - 128) - 0.391 * (self.u - 128)
        self.r = 1.164 * (self.y - 16) + 1.596 * (self.v - 128)
        return {"R": self.r, "G": self.g, "B": self.b}
