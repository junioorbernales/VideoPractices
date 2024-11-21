from app.services.conversions import Image

def test_rgb_to_yuv():
    img = Image(255, 0, 0)
    yuv = img.rgb_to_yuv()
    assert yuv["Y"] > 0

def test_yuv_to_rgb():
    img = Image(0, 0, 0)
    img.y, img.u, img.v = 16, 128, 128
    rgb = img.yuv_to_rgb()
    assert rgb["R"] > 0
