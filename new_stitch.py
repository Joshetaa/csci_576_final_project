#!/usr/bin/env python3
"""
Video-to-Panorama stitcher – translation-only with sliver concatenation

This pipeline uses block-based template matching to estimate pure integer-pixel
translations between consecutive frames, then appends only the new sliver of
pixels to a growing canvas. It assumes clips that pan without rotation or scale.

Usage
-----
python stitch.py --videos clip1.mp4 clip2.mp4 --stride 1 --out panorama.png

Dependencies
------------
* opencv-python
* numpy
"""

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def sample_frames(path: str, stride: int = 1) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {path}")
    idx, frames = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    print(f"{Path(path).name}: kept {len(frames)} frames (stride={stride})")
    return frames

# ---------------------------------------------------------------------------
# Motion estimation – template matching with grayscale & NCC
# ---------------------------------------------------------------------------
class MotionEstimator:
    """Estimate integer 2D translation between prev and cur via template matching."""
    def __init__(self, R: int = 1000, tpl_size: int = 200):
        self.R = R
        self.tpl_size = tpl_size

    def estimate(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        h, w = prev.shape[:2]
        # 1) center coordinates
        cx, cy = w // 2, h // 2
        half = self.tpl_size // 2

        # 2) define search window bounds
        x0 = max(cx - self.R - half, 0)
        y0 = max(cy - self.R - half, 0)
        x1 = min(cx + self.R + half, w)
        y1 = min(cy + self.R + half, h)

        # 3) extract patches
        tpl    = prev[cy-half:cy+half, cx-half:cx+half]
        sr     = cur[y0:y1, x0:x1]

        # 4) normalized cross-correlation (lighting-invariant)
        # Convert to grayscale for template matching
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        sr_gray = cv2.cvtColor(sr, cv2.COLOR_BGR2GRAY)
        
        res = cv2.matchTemplate(sr_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
         
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        hR, wR = res.shape
        if max_loc[0] in (0, wR-1) or max_loc[1] in (0, hR-1):
            print("⚠️  match hit the border—R may be too small or template ambiguous")
        # dx/dy relative to center
        dx = (max_loc[0] + x0) - (cx - half)
        dy = (max_loc[1] + y0) - (cy - half)
        # debug confidence
        # print(f"max_val={max_val:.3f}")

        # build forward translation matrix
        return np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float64)

    def create_debug_visualization(self, prev_frame: np.ndarray, cur_frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """Creates a side-by-side visualization of prev/cur frames with motion squares."""
        # --- Debug: draw comparison squares ---
        tpl_size = 16 # Keep fixed size for visualization clarity
        half = tpl_size // 2
        h_prev, w_prev = prev_frame.shape[:2]
        cx, cy = w_prev // 2, h_prev // 2

        # Create copy of previous frame and draw white square (template area)
        vis_prev = prev_frame.copy()
        prev_tl = (cx - half, cy - half) # Top-left
        prev_br = (cx + half, cy + half) # Bottom-right
        cv2.rectangle(vis_prev, prev_tl, prev_br, (255, 255, 255), 2) # White square

        # Create copy of current frame and draw red square (matched area)
        vis_cur = cur_frame.copy()
        # top-left corner of matched block in current frame using final dx, dy
        match_tl_x = int((cx - half) + dx)
        match_tl_y = int((cy - half) + dy)
        match_br_x = match_tl_x + tpl_size
        match_br_y = match_tl_y + tpl_size
        # Ensure coordinates are within bounds
        h_cur, w_cur = cur_frame.shape[:2]
        match_tl_x = max(0, match_tl_x)
        match_tl_y = max(0, match_tl_y)
        match_br_x = min(w_cur, match_br_x)
        match_br_y = min(h_cur, match_br_y)
        cv2.rectangle(vis_cur, (match_tl_x, match_tl_y), (match_br_x, match_br_y), (0, 0, 255), 2) # Red square

        # Combine images side-by-side
        if vis_prev.shape[0] != vis_cur.shape[0]:
            print(f"Warning: Frame heights differ for debug visualization {vis_prev.shape[0]} vs {vis_cur.shape[0]}!")
            # Return only current frame visualization as fallback
            return vis_cur
        else:
            combined_vis = np.hstack((vis_prev, vis_cur))
            return combined_vis
        # -------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Growable canvas – write-once sliver concatenation
# ---------------------------------------------------------------------------
class GrowableCanvas:
    def __init__(self, h: int, w: int):
        self.img = np.zeros((h, w, 3), np.uint8)
        self.mask = np.zeros((h, w), np.uint8)

    def _pad(self, y_min: int, x_min: int, y_max: int, x_max: int):
        top = max(-y_min, 0)
        bottom = max(y_max - self.img.shape[0], 0)
        left = max(-x_min, 0)
        right = max(x_max - self.img.shape[1], 0)
        if any((top, bottom, left, right)):
            self.img = np.pad(self.img,
                              ((top, bottom), (left, right), (0, 0)),
                              mode='constant', constant_values=0)
            self.mask = np.pad(self.mask,
                               ((top, bottom), (left, right)),
                               mode='constant', constant_values=0)
        return left, top

    def add(self, frame: np.ndarray, M_abs: np.ndarray):
        h, w = frame.shape[:2]
        x0 = int(np.round(M_abs[0, 2]))
        y0 = int(np.round(M_abs[1, 2]))
        y_min, x_min = y0, x0
        y_max, x_max = y0 + h, x0 + w
        dx, dy = self._pad(y_min, x_min, y_max, x_max)

        ox = x0 + dx
        oy = y0 + dy
        cslice = (slice(oy, oy + h), slice(ox, ox + w))

        empty = (self.mask[cslice] == 0)
        if not empty.any():
            return
        self.img[cslice][empty] = frame[empty]
        self.mask[cslice][empty] = 1

    def get_final_image(self):
        return np.clip(self.img, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Stitch orchestration
# ---------------------------------------------------------------------------
def stitch(videos: List[str], stride: int, out_path: str):
    frames = []
    for v in videos:
        frames.extend(sample_frames(v, stride))
    if not frames:
        raise ValueError("No frames extracted from inputs.")

    debug_dir = Path("out/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    for item in debug_dir.glob('*.png'):
        item.unlink()

    h, w = frames[0].shape[:2]
    canvas = GrowableCanvas(h, w)
    estimator = MotionEstimator()

    cum_dx = cum_dy = 0
    M_id = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    canvas.add(frames[0], M_id)
    cv2.imwrite(str(debug_dir / "debug_canvas_after_frame_000.png"), canvas.get_final_image())

    for i in range(1, len(frames)):
        M = estimator.estimate(frames[i-1], frames[i])
        dx, dy = M[0,2], M[1,2]
        combined_vis = estimator.create_debug_visualization(frames[i-1], frames[i], dx, dy)
        cv2.imwrite(str(debug_dir / f"debug_match_{i:03d}.png"), combined_vis)

        cum_dx += dx
        cum_dy += dy
        M_abs = np.array([[1, 0, cum_dx], [0, 1, cum_dy]], dtype=np.float64)
        print(f"Frame {i:03d}: shift=({dx:.0f},{dy:.0f}), cum=({cum_dx:.0f},{cum_dy:.0f})")
        canvas.add(frames[i], M_abs)
        cv2.imwrite(str(debug_dir / f"debug_canvas_after_frame_{i:03d}.png"), canvas.get_final_image())

    cv2.imwrite(out_path, cv2.cvtColor(canvas.img, cv2.COLOR_BGR2RGB))
    print(f"Panorama written to {out_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Panorama stitcher – sliver-based')
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--out',    required=True)
    args = ap.parse_args()
    for p in args.videos:
        if not Path(p).exists():
            ap.error(f"Video not found: {p}")
    stitch(args.videos, args.stride, args.out)
