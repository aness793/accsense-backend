from PIL import Image

img = Image.open(r"frontend\public\swin.png")
img.thumbnail((1920, 1920))  # keeps aspect ratio
img.save("swin.webp", "WEBP", lossless=True)