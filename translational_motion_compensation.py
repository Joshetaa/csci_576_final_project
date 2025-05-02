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

    def naive_estimate(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
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

        return translation_matrix

# ---------------------------------------------------------------------------
# Growable canvas - using warpAffine
# ---------------------------------------------------------------------------
class GrowableCanvas:
    def __init__(self, initial_frame: np.ndarray):
        h, w = initial_frame.shape[:2]
        # Store frames and their absolute transforms (frame 0 coords -> frame i coords)
        self.frames_data = []
        # Track absolute coordinate bounds (relative to frame 0 origin)
        self.xmin, self.ymin = 0.0, 0.0
        self.xmax, self.ymax = float(w), float(h)
        # Add the initial frame (identity transform relative to frame 0)
        # M_abs here transforms points FROM frame 0 TO frame 0 (identity)
        self.add(initial_frame, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64))

    def _update_bounds(self, M_abs: np.ndarray, h: int, w: int):
        # Calculate corners of the frame in the coordinate system of frame 0
        # M_abs transforms points FROM frame i TO frame 0
        # To find where frame i corners land in frame 0's system, we need inv(M_abs)
        # Let's redefine M_abs to be the transform FROM frame 0 TO frame i
        # This makes accumulation M_cum_i = M_rel @ M_cum_{i-1} simpler later

        corners = np.array([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h]
        ], dtype=np.float64).T # Shape (2, 4)

        # Apply transform M_abs (0 -> i) to origin corners (0,0) etc.
        # M_abs is 2x3. Add row [0,0,1] for 3x3 matrix mult with homogeneous coords
        M_abs_3x3 = np.vstack([M_abs, [0, 0, 1]])
        corners_hom = np.vstack([corners, np.ones((1, 4))]) # Shape (3, 4)

        transformed_corners = (M_abs_3x3 @ corners_hom)[:2, :] # Shape (2, 4)

        # Update bounds based on where the corners land
        self.xmin = min(self.xmin, np.min(transformed_corners[0, :]))
        self.ymin = min(self.ymin, np.min(transformed_corners[1, :]))
        self.xmax = max(self.xmax, np.max(transformed_corners[0, :]))
        self.ymax = max(self.ymax, np.max(transformed_corners[1, :]))

    def add(self, frame: np.ndarray, M_abs: np.ndarray):
        # M_abs: Transformation from frame 0 coordinate system to current frame's system
        h, w = frame.shape[:2]
        self._update_bounds(M_abs, h, w)
        # Store copies to avoid external modifications
        self.frames_data.append((frame.copy(), M_abs.copy()))

    def get_final_image(self):
        # Calculate final canvas size and the offset of frame 0's origin
        offset_x = -self.xmin
        offset_y = -self.ymin
        final_w = int(np.ceil(self.xmax + offset_x))
        final_h = int(np.ceil(self.ymax + offset_y))

        # Create final canvas and mask
        final_img = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        final_mask = np.zeros((final_h, final_w), dtype=np.uint8)

        print(f"Final Canvas Size: {final_w} x {final_h}")
        print(f"Offset (top-left of canvas relative to frame 0 origin): x={-offset_x:.2f}, y={-offset_y:.2f}")

        # Iterate through stored frames and warp them onto the canvas
        for frame, M_0_to_i in self.frames_data:
            # We need the transform from frame i coords TO final canvas coords
            # Final canvas coords = Frame 0 coords + (offset_x, offset_y)
            # Point_final = Point_frame0 + offset
            # Point_frame0 = M_0_to_i @ Point_frame_i (using 3x3)
            # So, Point_final = (M_0_to_i @ Point_frame_i) + offset

            # warpAffine needs M that transforms points FROM input (frame i) TO output (final canvas)
            # Let's construct M_i_to_final
            M_translate_offset = np.array([[1, 0, offset_x], [0, 1, offset_y]], dtype=np.float64)
            M_translate_3x3 = np.vstack([M_translate_offset, [0, 0, 1]])
            M_0_to_i_3x3 = np.vstack([M_0_to_i, [0, 0, 1]])

            # Transform: Frame i -> Frame 0 -> Final Canvas
            M_i_to_final_3x3 = M_translate_3x3 @ M_0_to_i_3x3
            M_i_to_final = M_i_to_final_3x3[:2, :] # Update M_cum for the current frame i

            # Warp the frame
            warped_frame = cv2.warpAffine(frame, M_i_to_final, (final_w, final_h),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

            # Warp a mask for the frame (use INTER_NEAREST for mask)
            frame_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255 # Mask covering the whole frame
            warped_mask = cv2.warpAffine(frame_mask, M_i_to_final, (final_w, final_h),
                                         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0 # Boolean mask

            # Find pixels to update (where final mask is empty and warped mask is valid)
            update_pixels = (final_mask == 0) & warped_mask

            # Update final image and mask
            final_img[update_pixels] = warped_frame[update_pixels]
            final_mask[update_pixels] = 1 # Mark as filled

        # Clip just in case, though should not be needed with uint8
        return np.clip(final_img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stitch orchestration
# ---------------------------------------------------------------------------
def stitch(videos: List[str], stride: int, estimator_arg: str, out_path: str):
    frames = []
    for v in videos:
        frames.extend(sample_frames(v, stride))
    if not frames:
        raise ValueError("No frames extracted from inputs.")

    debug_dir = Path("out/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    # Clear previous debug images if needed
    for item in debug_dir.glob('*.png'):
        item.unlink()

    # --- Initialize Canvas with the first frame ---
    canvas = GrowableCanvas(frames[0]) # Canvas adds frame 0 automatically
    estimator = MotionEstimator()

    # --- Initialize Cumulative Transform ---
    # M_cum transforms points FROM frame 0's coord system TO frame i's coord system. Start with identity for i=0.
    M_cum = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

    # --- Process subsequent frames ---
    for i in range(1, len(frames)):
        prev_frame = frames[i-1]
        cur_frame = frames[i]

        # Estimate relative motion M_rel (transform FROM prev frame coords TO cur frame coords)
        M_rel = None
        if estimator_arg == "feature_detection_and_matching":
             # Ensure this method returns the relative 2x3 matrix (prev -> cur)
             M_rel = estimator.feature_detection_and_matching(prev_frame, cur_frame, i-1, i, videos[0])
        elif estimator_arg == "naive":
             # estimate() returns M_rel (prev -> cur) directly now
             M_rel = estimator.naive_estimate(prev_frame, cur_frame)
        else:
            raise ValueError(f"Unknown estimator: {estimator_arg}")

        if M_rel is None:
             print(f"Warning: Skipping frame {i} due to estimation failure.")
             # Optionally: reuse previous M_rel or assume identity?
             continue # Simplest is to skip frame

        # Convert M_rel to 3x3 for matrix multiplication
        M_rel_3x3 = np.vstack([M_rel, [0, 0, 1]])

        # Update cumulative transform: M_cum_i = M_rel @ M_cum_{i-1}
        # Transforms points: Frame 0 -> Frame i-1 (via M_cum_{i-1}), then Frame i-1 -> Frame i (via M_rel)
        M_cum_3x3 = np.vstack([M_cum, [0, 0, 1]])
        M_new_cum_3x3 = M_rel_3x3 @ M_cum_3x3 # Note the order!
        M_cum = M_new_cum_3x3[:2, :] # Update M_cum for the current frame i

        # Add current frame to canvas with its absolute transform M_cum (0 -> i)
        canvas.add(cur_frame, M_cum)

        # --- Debugging Output ---
        dx_rel, dy_rel = M_rel[0, 2], M_rel[1, 2] # Relative shift for this step
        # Absolute position of current frame's origin (0,0) relative to frame 0's origin
        cdx_abs, cdy_abs = M_cum[0, 2], M_cum[1, 2]
        print(f"Frame {i:03d}: rel_shift=({dx_rel:+.1f},{dy_rel:+.1f}), abs_pos=({cdx_abs:+.1f},{cdy_abs:+.1f})") # Added + sign

        # Create and save debug visualization of relative motion if needed
        if i % 20 == 0:
            cv2.imwrite(str(debug_dir / f"debug_canvas_after_frame_{i:03d}.png"), canvas.get_final_image())


    # --- Final Output ---
    print("\nGenerating final canvas...")
    final_image = canvas.get_final_image()
    out_dir = Path("out")
    cv2.imwrite(str(out_dir / out_path), final_image)
    print(f"Stitched panorama saved to {out_dir / out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Panorama stitcher – sliver-based')
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--estimator', type=str, default='feature_detection_and_matching')
    args = ap.parse_args()
    ap.add_argument('--out', default=Path(args.videos[0]).stem + '.png')
    args = ap.parse_args()
    print("Creating output file: ", args.out)
    for p in args.videos:
        if not Path(p).exists():
            ap.error(f"Video not found: {p}")
    stitch(args.videos, args.stride, args.estimator, args.out)
