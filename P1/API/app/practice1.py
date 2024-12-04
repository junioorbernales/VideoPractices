from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import cv2
import subprocess
from subprocess import PIPE
import subprocess as sp
import shutil
from tempfile import NamedTemporaryFile
from scipy.fftpack import dct, idct
import pywt
from typing import List
import os
import logging
import ffmpeg
from pydantic import BaseModel
import json


app = FastAPI()

# Utility class for RGB/YUV conversion
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


# Zigzag mask for DCT
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


# DCT processing class
class DCT:
    def __init__(self, img):
        self.image = img

    def compressDCT(self, mask, N):
        self.image = np.float32(self.image)
        self.img_dct = np.zeros((self.image.shape[0] // N * N, self.image.shape[1] // N * N))
        for m in range(0, self.img_dct.shape[0], N):
            for n in range(0, self.img_dct.shape[1], N):
                block = self.image[m:m + N, n:n + N]
                coeff = cv2.dct(block)
                iblock = cv2.idct(coeff * mask)
                self.img_dct[m:m + N, n:n + N] = iblock

class DWT:
    def __init__(self, image):
        self.image = image

    def compressDWT(self, N):
        self.image = np.float32(self.image)
        self.img_dwt = np.zeros((self.image.shape[0] // N * N, self.image.shape[1] // N * N))
        for m in range(0, self.img_dwt.shape[0], N):
            for n in range(0, self.img_dwt.shape[1], N):
                block = self.image[m:m + N, n:n + N]
                coeffs = pywt.dwt2(block, 'db1')  # Perform DWT
                iblock = pywt.idwt2(coeffs, 'db1')  # Perform inverse DWT
                self.img_dwt[m:m + N, n:n + N] = iblock

# FastAPI endpoints

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI image processing app"}


@app.post("/convert-to-yuv/")
def convert_to_yuv(r: int, g: int, b: int):
    processor = Image(r, g, b)
    return processor.rgb_to_yuv()


@app.post("/convert-to-rgb/")
def convert_to_rgb(y: float, u: float, v: float):
    processor = Image(0, 0, 0)
    processor.y, processor.u, processor.v = y, u, v
    return processor.yuv_to_rgb()


@app.post("/resize-image/")
async def resize_image(file: UploadFile, width: int, height: int):
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save the uploaded file to the shared volume
        input_path = f"/shared/{file.filename}"
        output_path = f"/shared/resized_{file.filename}"
        with open(input_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Run FFmpeg inside the ffmpeg-docker container
        subprocess.run(
            [
                "docker", "exec", "api-ffmpeg-docker-1",
                "ffmpeg", "-i", input_path,
                "-vf", f"scale={width}:{height}",
                output_path
            ],
            check=True
        )

        # Return the resized image
        return FileResponse(output_path)

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")



@app.post("/compress-dct/")
async def compress_dct(file: UploadFile, c: int, n: int):
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = z_scan_mask(c, n)
        dct_processor = DCT(img)
        dct_processor.compressDCT(mask, n)
        compressed_img_path = f"/tmp/compressed_{file.filename}"
        cv2.imwrite(compressed_img_path, dct_processor.img_dct)
        return FileResponse(compressed_img_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-length-encode/")
def run_length_encode(bits_chain: List[int]):
    length = len(bits_chain)
    encoding = ""
    i = 0
    while i < length:
        count = 1
        while i + 1 < length and bits_chain[i] == bits_chain[i + 1]:
            count += 1
            i += 1
        encoding += str(bits_chain[i]) + str(count)
        i += 1
    return {"encoded": encoding}

@app.post("/compress-dwt/")
async def compress_dwt(file: UploadFile, n: int):
    """
    Compress an image using Discrete Wavelet Transform (DWT).
    """
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read the image from the uploaded file
        img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Apply DWT compression
        dwt_processor = DWT(img)
        dwt_processor.compressDWT(n)

        # Save the compressed image
        compressed_img_path = f"/tmp/compressed_dwt_{file.filename}"
        cv2.imwrite(compressed_img_path, dwt_processor.img_dwt)

        return FileResponse(compressed_img_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/resize-video/")
async def resize_video(
    file: UploadFile = File(...),
    width: int = 1280,
    height: int = 720
):
    # Validar el tipo de archivo
    if not file.filename.endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="El archivo debe ser un video (mp4, mov, avi, mkv)")

    try:
        # Save the uploaded file to the shared volume
        input_path = f"/shared/{file.filename}"
        output_path = f"/shared/resized_{file.filename}"
        with open(input_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Construir y ejecutar el comando ffmpeg
        command = [
            "docker", "exec", "api-ffmpeg-docker-1",
            "ffmpeg",
            "-i", input_path,
            "-vf", f"scale={width}:{height}",
            output_path
        ]
        subprocess.run(command, check=True)

        # Retornar el archivo procesado al cliente
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"resized_{file.filename}"
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el video: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error desconocido: {str(e)}")
    
@app.post("/modify-chroma-subsampling")
async def chroma_subsampling(file: UploadFile, request: str):
    if not file.filename.endswith((".mp4", ".mkv", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save the uploaded file to the shared volume
        input_path = f"/shared/{file.filename}"
        output_path = f"/shared/chroma_modified_{file.filename}"
        with open(input_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Run FFmpeg inside the ffmpeg-docker container
        subprocess.run(
            [
                "docker", "exec", "api-ffmpeg-docker-1",
                "ffmpeg", "-i", input_path,
                "-vf", f"format={request}",
                output_path
            ],
            check=True
        )

        # Return the resized image
        return FileResponse(output_path)

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")



@app.post("/video-info")
async def video_info(file: UploadFile):
    if not file.filename.endswith((".mp4", ".mkv", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="File must be a video")

    try:
        input_path = f"/shared/{file.filename}"
        with open(input_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        command = [
            "docker", "exec", "api-ffmpeg-docker-1",
            "ffprobe", "-v", "error",
            "-show_entries", "format:stream", 
            "-select_streams", "v:0",  
            "-print_format", "json",
            input_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        video_info = json.loads(result.stdout)

        format_info = video_info.get("format", {})
        stream_info = video_info.get("streams", [{}])[0] 

        response_data = {
            "filename": format_info.get("filename"),
            "duration": float(format_info.get("duration", 0)),
            "size": int(format_info.get("size", 0)),
            "bitrate": int(format_info.get("bit_rate", 0)),
            "width": stream_info.get("width"),
            "height": stream_info.get("height"),
            "codec": stream_info.get("codec_name"),
            "fps": eval(stream_info.get("avg_frame_rate", "0/1"))
        }

        return JSONResponse(content=response_data)

    except subprocess.CalledProcessError as e:
        logging.error(f"FFprobe failed: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"FFprobe error: {e.stderr}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
    
@app.post("/process-bbb")
async def process_bbb(file: UploadFile):
    if not file.filename.endswith((".mp4", ".mkv", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        #Paths
        input_path = f"/shared/{file.filename}"
        video_output_path = "/shared/bbb_20s.mp4"
        audio_aac_path = "/shared/bbb_20s_aac.m4a"
        audio_mp3_path = "/shared/bbb_20s_mp3.mp3"
        audio_ac3_path = "/shared/bbb_20s_ac3.ac3"
        packaged_output_path = "/shared/bbb_20s_packaged.mp4"

        #Save the uploaded file to the shared folder
        with open(input_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        #Step 1: Cut the video to 20 seconds
        subprocess.run(
            [
                "docker", "exec", "api-ffmpeg-docker-1",
                "ffmpeg", "-i", input_path,
                "-t", "20", "-c:v", "copy", "-c:a", "copy", video_output_path
            ],
            check=True
        )

        #Step 2: Export audio in AAC mono
        subprocess.run(
            [
                "docker", "exec", "api-ffmpeg-docker-1",
                "ffmpeg", "-i", video_output_path,
                "-vn", "-acodec", "aac", "-ac", "1", audio_aac_path
            ],
            check=True
        )

        #Step 3: Export audio in MP3 stereo with lower bitrate
        subprocess.run(
            [
                "docker", "exec", "api-ffmpeg-docker-1",
                "ffmpeg", "-i", video_output_path,
                "-vn", "-acodec", "libmp3lame", "-ac", "2", "-b:a", "96k", audio_mp3_path
            ],
            check=True
        )

        #Step 4: Export audio in AC3 codec
        subprocess.run(
            [
                "docker", "exec", "api-ffmpeg-docker-1",
                "ffmpeg", "-i", video_output_path,
                "-vn", "-acodec", "ac3", audio_ac3_path
            ],
            check=True
        )

        #Step 5: Package everything into a single MP4
        subprocess.run(
            [
                "docker", "exec", "api-ffmpeg-docker-1",
                "ffmpeg", "-i", video_output_path,
                "-i", audio_aac_path, "-i", audio_mp3_path, "-i", audio_ac3_path,
                "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0", "-map", "3:a:0",
                "-c:v", "copy", "-c:a", "copy", packaged_output_path
            ],
            check=True
        )

        #Return the packaged file
        return FileResponse(packaged_output_path, media_type="video/mp4", filename="bbb_20s_packaged.mp4")

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg command failed: {e.stderr}")
        raise HTTPException(status_code=500, detail="FFmpeg processing error.")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
