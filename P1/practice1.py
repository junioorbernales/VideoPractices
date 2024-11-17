'''
%pip3 install fastapi
%pip3 install "uvicorn[standard]" #Servidor virtual para usar FastAPI
uvicorn P1.practice1:app --reload para encender el servidor
'''

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import ffmpeg
import numpy as np
import cv2
from scipy.fftpack import dct, idct
import skimage.io as io
import pywt
from typing import List
import os

app = FastAPI()

# Utility classes and functions
class ImageProcessor:
    def __init__(self, r, g, b):
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

# Zigzag mask
def z_scan_mask(C, N):
    mask = np.zeros((N, N))
    mask_m, mask_n = 0, 0
    for i in range(C):
        if i == 0:
            mask[mask_m, mask_n] = 1
        else:
            if (mask_m + mask_n) % 2 == 0:  # Even, move up-right
                mask_m -= 1
                mask_n += 1
                if mask_m < 0:
                    mask_m += 1
                if mask_n >= N:
                    mask_n -= 1
            else:  # Odd, move down-left
                mask_m += 1
                mask_n -= 1
                if mask_m >= N:
                    mask_m -= 1
                if mask_n < 0:
                    mask_n += 1
            mask[mask_m, mask_n] = 1
    return mask

# DCT class
class DCT:
    def __init__(self, img):
        self.image = img

    def compress_dct(self, mask, N):
        self.image = np.float32(self.image)
        self.img_dct = np.zeros((self.image.shape[0] // N * N, self.image.shape[1] // N * N))
        for m in range(0, self.img_dct.shape[0], N):
            for n in range(0, self.img_dct.shape[1], N):
                block = self.image[m:m + N, n:n + N]
                coeff = cv2.dct(block)
                iblock = cv2.idct(coeff * mask)
                self.img_dct[m:m + N, n:n + N] = iblock

# Routes
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI image processing app"}

@app.post("/convert-to-yuv/")
def convert_to_yuv(r: int, g: int, b: int):
    processor = ImageProcessor(r, g, b)
    return processor.rgb_to_yuv()

@app.post("/convert-to-rgb/")
def convert_to_rgb(y: float, u: float, v: float):
    processor = ImageProcessor(0, 0, 0)
    processor.y, processor.u, processor.v = y, u, v
    return processor.yuv_to_rgb()

@app.post("/resize-image/")
async def resize_image(file: UploadFile, width: int, height: int):
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="File must be an image")
    output_path = f"resized_{file.filename}"
    try:
        (
            ffmpeg
            .input(file.file)
            .filter("scale", width, height)
            .output(output_path)
            .run()
        )
        return FileResponse(output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compress-dct/")
async def compress_dct(file: UploadFile, c: int, n: int):
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        img = io.imread(file.file, as_gray=True)
        mask = z_scan_mask(c, n)
        dct_processor = DCT(img)
        dct_processor.compress_dct(mask, n)
        compressed_img_path = f"compressed_{file.filename}"
        cv2.imwrite(compressed_img_path, dct_processor.img_dct)
        return FileResponse(compressed_img_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-length-encode/")
def run_length_encode(bits_chain: List[int]):
    encoding = ""
    i = 0
    length = len(bits_chain)
    while i < length:
        count = 1
        while i + 1 < length and bits_chain[i] == bits_chain[i + 1]:
            count += 1
            i += 1
        encoding += str(bits_chain[i]) + str(count)
        i += 1
    return {"encoded": encoding}
