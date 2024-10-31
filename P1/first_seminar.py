import ffmpeg

class RGB:

    y = 0
    u = 0
    v = 0

    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b

    def rgb_to_yuv(self):
        self.y = (0.257 * self.r) + (0.504 * self.g) + (0.098 * self.b) + 16
        self.u = (-0.148 * self.r) - (0.291 * self.g) + (0.439 * self.b) + 128
        self.v = (0.439 * self.r) - (0.368 * self.g) - (0.071 * self.b) + 128
    
    def yuv_to_rgb(self):
        self.b = 1.164 * (self.y - 16) + 2.018 * (self.u - 128)
        self.g = 1.164 * (self.y - 16) - 0.813 * (self.v - 128) - 0.391 * (self.u - 128)
        self.r = 1.164 * (self.y - 16) + 1.596 * (self.v - 128) 
    
    def print_RGB_values(self):
        print("R:", self.r, "\n")
        print("G:", self.g, "\n")
        print("B:", self.b, "\n")

    def print_YUV_values(self):
        print("Y:", self.y, "\n")
        print("U:", self.u, "\n")
        print("V:", self.v, "\n")
    
i1 = RGB(100, 50, 255)
i1.print_RGB_values()
i1.rgb_to_yuv()
i1.print_YUV_values()
i1.yuv_to_rgb()
i1.print_RGB_values()

image = ffmpeg.input('image.jpg')
resize_image = image.output(image, 'output.jpg')
ffmpeg.run(image)
