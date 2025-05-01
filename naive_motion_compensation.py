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
# Motion estimation – quadrant-based phase correlation
# ---------------------------------------------------------------------------
class MotionEstimator:
    """Estimate integer 2D translation between prev and cur via phase correlation or feature matching."""
    def __init__(self, min_match_count=10):
        # Parameters for feature matching (ORB + RANSAC)
        self.orb = cv2.ORB_create(nfeatures=5000) # Use more features potentially
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.min_match_count = min_match_count
        # No parameters needed for phase correlation itself

    def estimate(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        h, w = prev.shape[:2]
        prev_gray_orb = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) # uint8 for ORB
        cur_gray_orb = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        # --- Attempt 1: Feature Matching (ORB + RANSAC) ---
        kp1, des1 = self.orb.detectAndCompute(prev_gray_orb, None)
        kp2, des2 = self.orb.detectAndCompute(cur_gray_orb, None)

        M = None
        inlier_count = 0
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            matches = self.bf.knnMatch(des1, des2, k=2)
            good_matches = []
            # Ratio test
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) >= 4: # Min points for estimateAffinePartial2D
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

                if M is not None and mask is not None:
                    inlier_count = np.sum(mask)

        # --- Decision: Use Feature Match or Fallback? ---
        if M is not None and inlier_count >= self.min_match_count:
            # Use feature-based estimate
            print(f"    -> Using ORB (Inliers: {inlier_count})")
            dx, dy = M[0, 2], M[1, 2]
            # print(f"    -> Using ORB: dx={dx:.1f}, dy={dy:.1f}, inliers={inlier_count}") # Optional debug
            # RANSAC gives transform from src (prev) to dst (cur).
            # Need to negate for accumulation loop expectation.
            return np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float64)
        else:
            # --- Attempt 2: Fallback to Phase Correlation (9-block weighted) ---
            print("    -> Using Phase Correlation Fallback")
            # print("    -> Using Phase Correlation Fallback") # Optional debug
            prev_gray = prev_gray_orb.astype(np.float32) # Convert uint8 gray to float32
            cur_gray = cur_gray_orb.astype(np.float32)

            # Define 9 block centers for a 3x3 grid
            # Block size will be roughly h/3 x w/3
            # Centers at (w/6, h/6), (w/2, h/6), (5w/6, h/6), ... (w/2, h/2), ... (5w/6, 5h/6)
            block_h, block_w = h // 6, w // 6 # Use smaller blocks for 3x3 grid
            half_bh, half_bw = block_h // 2, block_w // 2

            centers = []
            for row in range(3):
                cy = (2 * row + 1) * h // 6
                for col in range(3):
                    cx = (2 * col + 1) * w // 6
                    centers.append((cx, cy))

            shifts = []
            variances = []
            epsilon = 1e-6 # To avoid division by zero

            for cx, cy in centers:
                # Define block boundaries
                y_start, y_end = cy - half_bh, cy + half_bh
                x_start, x_end = cx - half_bw, cx + half_bw

                # Ensure block is within image bounds (can happen with small images)
                y_start, y_end = max(0, y_start), min(h, y_end)
                x_start, x_end = max(0, x_start), min(w, x_end)

                if (y_end - y_start) <= 0 or (x_end - x_start) <= 0:
                    continue # Skip if block has zero size

                prev_block = prev_gray[y_start:y_end, x_start:x_end]
                cur_block = cur_gray[y_start:y_end, x_start:x_end]

                # Calculate variance of the previous block
                variance = np.var(prev_block)
                variances.append(variance + epsilon)

                # Phase correlation for the block
                block_h_actual, block_w_actual = prev_block.shape
                win = cv2.createHanningWindow((block_w_actual, block_h_actual), cv2.CV_32F)
                prev_block_win = prev_block * win
                cur_block_win = cur_block * win

                shift, response = cv2.phaseCorrelate(prev_block_win, cur_block_win)
                shifts.append(shift) # (dx, dy)

            if not shifts or sum(variances) == 0:
                 # Handle cases with no valid blocks or zero variance (e.g., black frames)
                final_dx, final_dy = 0.0, 0.0
            else:
                # Weighted average
                total_variance = sum(variances)
                final_dx = sum(s[0] * v for s, v in zip(shifts, variances)) / total_variance
                final_dy = sum(s[1] * v for s, v in zip(shifts, variances)) / total_variance

            # Phase correlate gives (dx, dy) shift from prev to cur.
            # The matrix should reflect this forward motion for accumulation.
            # However, based on previous runs and the template matching logic,
            # the accumulation loop seems to expect the *negated* shift.
            return np.array([[1, 0, -final_dx], [0, 1, -final_dy]], dtype=np.float64)

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


    def feature_detection_and_matching(self, frame1, frame2, frame_idx1, frame_idx2, video_file):
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Detect keypoints and compute descriptors in both frames
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            print("Descriptors not found in one of the frames.")
            return None

        # Use BFMatcher to match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if not matches:
            print("No matches found.")
            return None

        # Compute all dx, dy from matched keypoints
        dxs, dys = [], []
        for match in matches:
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            dxs.append(pt2[0] - pt1[0])
            dys.append(pt2[1] - pt1[1])

        if dxs and dys:
            # Compute median dx, dy
            median_dx = np.median(dxs)
            median_dy = np.median(dys)
        else:
            median_dx, median_dy = 0.0, 0.0

        # Round the median dx and dy for translation matrix
        median_dx = round(median_dx)
        median_dy = round(median_dy)

        # Create affine translation matrix using rounded values
        translation_matrix = np.array([
            [1, 0, -median_dx],
            [0, 1, -median_dy]
        ], dtype=np.float32)

        print(f"\n{video_file}: Estimated translation from Frame {frame_idx1} to Frame {frame_idx2}")
        print(f"dx = {median_dx}, dy = {median_dy}")
        print(f"Affine translation matrix:\n{translation_matrix}")

        # Concatenate images side by side for visualization
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        canvas_height = max(h1, h2)
        canvas_width = w1 + w2
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:h1, :w1] = frame1
        canvas[:h2, w1:w1 + w2] = frame2

        # Font for labels
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw the matches and labels on the canvas
        for i, match in enumerate(matches):
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            pt2 = tuple(map(int, kp2[match.trainIdx].pt))
            pt2_shifted = (int(pt2[0] + w1), int(pt2[1]))   # Shift second point to right image

            # Draw a line between matched keypoints
            color = (0, 255, 0)
            cv2.line(canvas, pt1, pt2_shifted, color, 2)

        # Put match number text near the middle of the line
        # label_pos = ((pt1[0] + pt2_shifted[0]) // 2, (pt1[1] + pt2_shifted[1]) // 2 - 10)
        # cv2.putText(canvas, f"Match {i + 1}", label_pos, font, 0.5, (0, 255, 255), 1)

        # Add frame indices and video file name
        cv2.putText(canvas, f"Frame {frame_idx1}", (10, 30), font, 1, (255, 0, 0), 2)
        cv2.putText(canvas, f"Frame {frame_idx2}", (w1 + 10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(canvas, f"Video: {video_file}", (10, 60), font, 0.8, (0, 255, 255), 2)

        # Print coordinates, deltas, and responses
        print(f"\nMatched keypoints between Frame {frame_idx1} and Frame {frame_idx2}:")
        # for i, match in enumerate(matches):
        #     pt1 = kp1[match.queryIdx].pt
        #     pt2 = kp2[match.trainIdx].pt
        #     x1, y1 = round(pt1[0]), round(pt1[1])
        #     x2, y2 = round(pt2[0]), round(pt2[1])
        #     dx = round(x2 - x1)
        #     dy = round(y2 - y1)
        #     response = kp1[match.queryIdx].response
        #     print(f"Match {i + 1}:")
        #     print(f"  Frame {frame_idx1} -> ({x1}, {y1})")
        #     print(f"  Frame {frame_idx2} -> ({x2}, {y2})")
        #     print(f"  Delta: (dx, dy) = ({dx}, {dy})")
        #     print(f"  Response: {response:.2e}")

        return translation_matrix







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
        # M = estimator.estimate(frames[i-1], frames[i])
        M = estimator.feature_detection_and_matching(frames[i-1], frames[i], i-1, i, videos[0])
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
