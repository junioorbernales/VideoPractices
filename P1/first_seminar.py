import ffmpeg
import numpy as np
import os

class Image:

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
    
#Exercise 2
i1 = Image(100, 50, 255)
i1.print_RGB_values()
i1.rgb_to_yuv()
i1.print_YUV_values()
i1.yuv_to_rgb()
i1.print_RGB_values()

'''
#Exercise 3
image = ffmpeg.input('image.jpg')
resize_image = image.output(image, 'output.jpg')
ffmpeg.run(image)
'''

#Exercise 4
#Code taken from (https://stackoverflow.com/questions/57366966/serpentine-scan-pattern-generator)
def zigzag(dims):
    r = np.arange(np.prod(dims))
    out = []
    for d in dims:
        out.append(np.abs((1|((d+r)<<1))%(d<<2)-(d<<1))>>1)
        r //= d
    return np.transpose(out[::-1])
###

path = 'coltrane.jpg'
coltrane = ffmpeg.input(path)

try:
    probe = ffmpeg.probe(path)
except ffmpeg.Error as e:
    print("stdout:", e.stdout.decode('utf-8'))
    print("stderr:", e.stderr.decode('utf-8'))

'''
image_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')

width = int(coltrane['width'])
height = int(coltrane['height'])
coltrane_array = (
    np
    .frombuffer(coltrane, np.uint8)
    .reshape([-1, height, width, 3])
)
'''