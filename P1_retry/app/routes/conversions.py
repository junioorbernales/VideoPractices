from fastapi import APIRouter
from app.services.conversions import Image

router = APIRouter()

@router.post("/to-yuv")
def convert_to_yuv(r: int, g: int, b: int):
    processor = Image(r, g, b)
    return processor.rgb_to_yuv()

@router.post("/to-rgb")
def convert_to_rgb(y: float, u: float, v: float):
    processor = Image(0, 0, 0)
    processor.y, processor.u, processor.v = y, u, v
    return processor.yuv_to_rgb()
