import shutil
import subprocess
import sys
from pathlib import Path

# If "ffmpeg" isn't in your Windows PATH, replace "ffmpeg" with the full path,
# e.g. r"C:\ffmpeg\bin\ffmpeg.exe"
FFMPEG_BIN = "ffmpeg"

# Resolves automatically to the folder where convert.py lives (.../accsense-frontend/public)
BASE_DIR = Path(__file__).resolve().parent

VIDEO_FILES = [

    "major.mp4"
   # replace with your 4th filename
]

def fix_video(filename: str):
    input_path = BASE_DIR / filename

    if not input_path.exists():
        print(f"[SKIPPED] File not found: {input_path}")
        return

    output_path = input_path.with_name(f"{input_path.stem}_web.mp4")

    command = [
        FFMPEG_BIN,
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level:v", "4.1",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path)
    ]

    print(f"Converting: {input_path.name} -> {output_path.name}...")

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        print(f"[OK] Saved to: {output_path}\n")
    else:
        print(f"[ERROR] Failed on {input_path.name}:\n{result.stderr}\n")

if __name__ == "__main__":
    if not shutil.which(FFMPEG_BIN) and not Path(FFMPEG_BIN).exists():
        print(f"[FATAL] '{FFMPEG_BIN}' not found. Install FFmpeg or set FFMPEG_BIN to the full path of ffmpeg.exe.")
        sys.exit(1)

    for video in VIDEO_FILES:
        fix_video(video)