import cv2

# ←––––– CONFIG ––––––→
VIDEO_PATH  = "real_001.mp4"    # your input video
FRAME_STEP  = 20                # sample every Nth frame
OUTPUT_PATH = "final_still.jpg" # output image filename
# ←–––––––––––––––––––––→

# 1) Load & subsample frames
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_PATH}")

frames = []
idx = 0
print(f"Sampling every {FRAME_STEP}th frame…")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if idx % FRAME_STEP == 0:
        frames.append(frame)
        h, w = frame.shape[:2]
        print(f" • frame #{idx}: {w}×{h}")
    idx += 1
cap.release()

if len(frames) < 2:
    raise RuntimeError("Need at least two frames to stitch!")

# 2) Create & configure Stitcher
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
stitcher.setWaveCorrection(True)       # straighten horizons
stitcher.setPanoConfidenceThresh(0.6)  # drop low-confidence overlaps

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
