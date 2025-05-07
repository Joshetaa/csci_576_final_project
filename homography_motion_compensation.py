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

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        # Keep a reasonable number of good matches
        num_good_matches = min(50, int(len(matches) * 0.8)) # Example: Keep the best 50 or 80%
        good_matches = matches[:num_good_matches]

        if len(good_matches) < 4: # Need at least 4 points for homography
            print(f"Not enough good matches found between frame {frame_idx1} and {frame_idx2} ({len(good_matches)} found).")
            return None

        # Extract location of good matches
        points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography using RANSAC
        # --- old ---
        # H_rel, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
        # if H_rel is None:
        #     print(f"Homography estimation failed between frame {frame_idx1} and {frame_idx2}.")
        #     return None

        # --- new ---
        H_affine, mask = cv2.estimateAffinePartial2D(
        points1, points2,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=5000,
        confidence=0.995)

        if H_affine is None:
            print(f"Affine estimation failed between frames {frame_idx1}->{frame_idx2}")
            return None
        # -------------------
        
        # promote 2×3 to 3×3 so the rest of the pipeline is unchanged
        H_rel = np.vstack([H_affine, [0, 0, 1]]).astype(np.float64)

        # Optional: Check the number of inliers
        if mask is not None:
             num_inliers = np.sum(mask)
             if num_inliers < 4:
                 print(f"Homography found, but too few inliers ({num_inliers}) between frame {frame_idx1} and {frame_idx2}.")
                 return None # Treat as failure
        else: # Should not happen if H_rel is not None, but check anyway
             print(f"Homography estimated but RANSAC mask is None between frame {frame_idx1} and {frame_idx2}.")
             return None

        return H_rel # Return the 3x3 homography matrix


class GrowableCanvas:
    """Manages the growing canvas for stitching frames."""

    def __init__(self):
        self.frames_data: List[Tuple[np.ndarray, np.ndarray]] = []  # Store (frame, H_cum)
        self.min_x, self.min_y = 0.0, 0.0
        self.max_x, self.max_y = 0.0, 0.0
        self.initialized = False # Added flag

    def _update_bounds(self, frame: np.ndarray, H_cum: np.ndarray): # Changed M_abs to H_cum
        """Update canvas bounds using perspective transform on corners.""" # Changed docstring
        h, w = frame.shape[:2]
        corners = np.array([
            [0, 0], [w, 0], [w, h], [0, h]
        ], dtype=np.float32)
        corners_reshaped = corners.reshape(-1, 1, 2) # Reshape for perspectiveTransform

        # Apply the cumulative homography H_cum (must be 3x3)
        if H_cum is None or H_cum.shape != (3, 3):
             print("Warning: Invalid H_cum matrix in _update_bounds. Skipping bounds update.")
             transformed_corners = corners_reshaped # Use original corners if transform invalid
        else:
            try:
                # Use perspectiveTransform instead of transform
                transformed_corners = cv2.perspectiveTransform(corners_reshaped, H_cum)
                if transformed_corners is None: # perspectiveTransform can fail
                    print("Warning: cv2.perspectiveTransform returned None. Using original corners.")
                    transformed_corners = corners_reshaped
            except cv2.error as e:
                print(f"Error during perspectiveTransform: {e}. Using original corners.")
                transformed_corners = corners_reshaped

        # Extract x, y coordinates.
        cx = transformed_corners[:, 0, 0]
        cy = transformed_corners[:, 0, 1]

        current_min_x = np.min(cx)
        current_min_y = np.min(cy)
        current_max_x = np.max(cx)
        current_max_y = np.max(cy)

        if not self.initialized:  # Use initialized flag
            self.min_x, self.min_y = current_min_x, current_min_y
            self.max_x, self.max_y = current_max_x, current_max_y
            self.initialized = True # Set flag on first successful update
        else:
            self.min_x = min(self.min_x, current_min_x)
            self.min_y = min(self.min_y, current_min_y)
            self.max_x = max(self.max_x, current_max_x)
            self.max_y = max(self.max_y, current_max_y)

    def add(self, frame: np.ndarray, H_cum: np.ndarray): # Changed M_abs to H_cum
        """Add a frame and its cumulative homography matrix."""
        if H_cum is not None and H_cum.shape == (3, 3):
            # Update bounds *before* adding, so initialization flag works correctly
            self._update_bounds(frame, H_cum)
            self.frames_data.append((frame.copy(), H_cum.copy()))
        else:
             print("Warning: Attempted to add frame with invalid H_cum. Frame skipped.")


    def get_final_image(self) -> np.ndarray:
        """Create the final stitched panorama using perspective warping."""
        if not self.frames_data:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # Calculate the size of the final canvas
        final_w = int(np.ceil(self.max_x - self.min_x))
        final_h = int(np.ceil(self.max_y - self.min_y))
        final_w = max(final_w, 1) # Ensure minimum size
        final_h = max(final_h, 1)

        # Translation matrix (3x3) to shift to origin
        T = np.array([
            [1, 0, -self.min_x],
            [0, 1, -self.min_y],
            [0, 0, 1] # Make it 3x3
        ], dtype=np.float64)

        canvas = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        print(f"Creating final canvas of size: {final_w}x{final_h}")
        print(f"Canvas bounds: x=[{self.min_x:.2f}, {self.max_x:.2f}], y=[{self.min_y:.2f}, {self.max_y:.2f}]")

        for i, (frame, H_cum) in enumerate(self.frames_data): # Use H_cum
            print(f"Processing frame {i+1}/{len(self.frames_data)} for final warp...")

            # Combine H_cum with canvas translation T
            H_final = T @ H_cum # 3x3 multiplication

            try:
                # Use warpPerspective instead of warpAffine
                warped_frame = cv2.warpPerspective(frame, H_final, (final_w, final_h),
                                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

                # Update canvas only where it's currently black (empty)
                # and the warped_frame has non-black content.
                # This prevents overwriting already filled parts of the canvas
                # and helps avoid issues where black borders of a new frame
                # would effectively erase existing content or leave black lines.
                canvas_is_empty_mask = (np.sum(canvas, axis=2) == 0)
                warped_frame_has_content_mask = (np.sum(warped_frame, axis=2) > 0)
                update_mask = canvas_is_empty_mask & warped_frame_has_content_mask
                canvas[update_mask] = warped_frame[update_mask]

            except cv2.error as e:
                print(f"Error warping frame {i} with warpPerspective: {e}")
                # Print relevant info for debugging
                h_f, w_f = frame.shape[:2]
                print(f"Frame shape: {h_f}x{w_f}, H_final: {H_final}")
                continue

        return canvas


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
    canvas = GrowableCanvas()
    estimator = MotionEstimator()

    # --- Initialize Cumulative Transform ---
    # H_cum transforms points FROM frame 0's coord system TO frame i's coord system. Start with identity for i=0.
    H_cum = np.eye(3, dtype=np.float64) # Use 3x3 Identity for H_cum
    prev_frame = frames[0]
    canvas.add(prev_frame, H_cum) # Add first frame with identity H_cum

    # --- Process subsequent frames ---
    for i in range(1, len(frames)):
        cur_frame = frames[i]

        # Estimate relative motion (Homography)
        H_rel = None
        if estimator_arg == "feature_detection_and_matching":
             # Ensure this method returns the relative 3x3 matrix (prev -> cur)
             H_rel = estimator.feature_detection_and_matching(prev_frame, cur_frame, i-1, i, videos[0])
        elif estimator_arg == "naive":
             # estimate() returns M_rel (prev -> cur) directly now
             H_rel = estimator.naive_estimate(prev_frame, cur_frame)
        else:
            raise ValueError(f"Unknown estimator: {estimator_arg}")

        if H_rel is None:
             print(f"Motion estimation failed between frame {i-1} and {i}. Skipping frame {i}.")
             # Optionally: reuse previous H_rel or assume identity?
             continue # Simplest is to skip frame

        # Accumulate transformations (3x3 Matrix multiplication)
        H_rel_inv = np.linalg.inv(H_rel)
        H_cum = H_rel_inv @ H_cum
        H_cum /= H_cum[2, 2]  # Normalize to keep numerical stability

        print(f"Adding frame {i} to canvas.")
        canvas.add(cur_frame, H_cum) # Add with the new H_cum

        if i % 100 == 0:
            print(f"Frame {i}: |H_cum| = {np.linalg.det(H_cum):.4f}, bottom row = {H_cum[2]}") 
            cv2.imwrite(str(debug_dir / f"debug_canvas_after_frame_{i:03d}.png"), canvas.get_final_image())


        # Update for next iteration
        prev_frame = cur_frame

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