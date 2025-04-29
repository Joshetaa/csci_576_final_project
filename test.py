import cv2
import numpy as np

# ←––––– CONFIG ––––––→
VIDEO_PATH  = "real_001.mp4"     # input video
FRAME_STEP  = 10                 # sample every Nth frame
OUTPUT_PATH = "final_still.jpg"     # output image filename
SCALE       = 1                     # downscale frames for speed/memory
DIFF_THRESH = 50                    # % of pixels that must change to count as "different"
BLUR_KERNEL = (5, 5)                 # smoothing for noise reduction
PIXEL_THRESH = 10                   # pixel difference threshold for significant change
# ←–––––––––––––––––––––→

def is_significantly_different(img1, img2, threshold=DIFF_THRESH):
    gray1 = cv2.GaussianBlur(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), BLUR_KERNEL, 0)
    gray2 = cv2.GaussianBlur(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), BLUR_KERNEL, 0)

    diff = cv2.absdiff(gray1, gray2)
    mask = (diff > PIXEL_THRESH).astype(np.uint8)
    changed = cv2.countNonZero(mask)
    percent_diff = changed / (img1.shape[0] * img1.shape[1]) * 100

    return percent_diff > threshold

# 1) Load & subsample frames
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_PATH}")

frames = []
prev_frame = None
idx = 0
print(f"Sampling every {FRAME_STEP}th frame with filtering…")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if idx % FRAME_STEP == 0:
        frame = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        if prev_frame is None or is_significantly_different(prev_frame, frame):
            frames.append(frame)
            prev_frame = frame
            print(f" • frame #{idx} added")
        else:
            print(f" • frame #{idx} skipped (too similar)")
    idx += 1

cap.release()

if len(frames) < 2:
    raise RuntimeError("Need at least two significantly different frames to stitch!")

breakpoint()
# 2) Create & configure Stitcher
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
stitcher.setWaveCorrection(True)
stitcher.setPanoConfidenceThresh(0.6)

print("Running high-quality panorama stitch… this can take a minute")
status, pano = stitcher.stitch(frames)
if status != cv2.Stitcher_OK:
    raise RuntimeError(f"Stitcher failed (code {status})")

# 3) Crop any black borders
gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(cnts[0])
final = pano[y:y+h, x:x+w]

# 4) Save result
cv2.imwrite(OUTPUT_PATH, final)
print(f"✅ Saved high-quality still as {OUTPUT_PATH} ({final.shape[1]}×{final.shape[0]})")
