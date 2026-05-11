import cv2

def get_video_codec_opencv(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return "Error: Cannot open video"
    
    # Get FourCC code
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    cap.release()
    return codec

# Usage
codec = get_video_codec_opencv("major.mp4")
print(f"Codec FourCC: {codec}")

# FourCC reference:
# avc1, H264 = H.264
# hev1, hvc1 = H.265/HEVC
# mp4v = MPEG-4