class RGB:
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b
    
    def rgb_to_yuv(self):
        Y = (0,257 * self.r) + (0,504 * self.g) + (0,098 * self.b) + 16
        U = (-0,148 * self.r) - (0,291 * self.g) + (0,439 * self.b) + 128
        V = (0,439 * self.r) - (0,368 * self.g) - (0,071 * self.b) + 128 
        return Y, U, V