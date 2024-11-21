from fastapi import FastAPI
from app.routes import conversions, image_resize, compression, run_length

app = FastAPI()

# Include routes
app.include_router(conversions.router, prefix="/conversions", tags=["Color Conversions"])
app.include_router(image_resize.router, prefix="/resize", tags=["Image Resize"])
app.include_router(compression.router, prefix="/compression", tags=["Compression"])
app.include_router(run_length.router, prefix="/run-length", tags=["Run Length Encoding"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI Image Processing API"}
