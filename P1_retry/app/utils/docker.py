import subprocess
from fastapi import HTTPException

def run_ffmpeg(input_path: str, output_path: str, scale: str):
    try:
        subprocess.run(
            [
                "docker", "exec", "ffmpeg-docker",
                "ffmpeg", "-i", input_path,
                "-vf", f"scale={scale}",
                output_path
            ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg error: {str(e)}")
