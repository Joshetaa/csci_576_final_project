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
from typing import List, Tuple, Optional

import cv2
import numpy as np

# added
from itertools import permutations
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

    def __init__(self, initial_frame: np.ndarray = None):
        # Store (frame, H_cum) tuples
        self.frames_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self.min_x = self.min_y = 0.0
        self.max_x = self.max_y = 0.0
        self.initialized = False

        # If user passed an initial frame, add it with identity homography
        if initial_frame is not None:
            H_identity = np.eye(3, dtype=np.float64)
            self.add(initial_frame, H_identity)

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
                                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                # Simple Overwrite Blending (remains the same logic for now)
                mask = np.sum(warped_frame, axis=2) > 0
                canvas[mask] = warped_frame[mask]

            except cv2.error as e:
                print(f"Error warping frame {i} with warpPerspective: {e}")
                # Print relevant info for debugging
                h_f, w_f = frame.shape[:2]
                print(f"Frame shape: {h_f}x{w_f}, H_final: {H_final}")
                continue

        return canvas

def find_best_overlap(prev_frames: List[np.ndarray],
                      new_frames:  List[np.ndarray],
                      estimator:   MotionEstimator,
                      hist_thresh: float = 0.5
                      ) -> Tuple[int, int, np.ndarray]:
    """
    Fast O(N²) overlap finder with descriptor‑caching and histogram pruning.
    Compares all prev_frames × new_frames, but skips most pairs via cheap hist tests.
    """
    n_prev, n_curr = len(prev_frames), len(new_frames)
    if n_prev == 0 or n_curr == 0:
        raise ValueError("Empty frame list!")

    # 1) Precompute ORB features (gray, kp, des) for every frame
    prev_feats = []
    for f in prev_frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        kp, des = estimator.orb.detectAndCompute(g, None)
        prev_feats.append((g, kp, des))

    curr_feats = []
    for f in new_frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        kp, des = estimator.orb.detectAndCompute(g, None)
        curr_feats.append((g, kp, des))

    # 2) Precompute tiny histograms for pruning
    def tiny_hist(gray):
        small = cv2.resize(gray, (64,64))
        h = cv2.calcHist([small], [0], None, [32], [0,256])
        return cv2.normalize(h, h).flatten()

    prev_hists = [tiny_hist(g) for g,_,_ in prev_feats]
    curr_hists = [tiny_hist(g) for g,_,_ in curr_feats]

    best_i = best_j = -1
    best_H = None
    best_inliers = 0

    print(f"[DEBUG] Matching all {n_prev} prev‑frames × {n_curr} new‑frames")

    for i, (g1, kp1, des1) in enumerate(prev_feats):
        if des1 is None:
            continue
        h1 = prev_hists[i]

        for j, (g2, kp2, des2) in enumerate(curr_feats):
            if des2 is None:
                continue
            h2 = curr_hists[j]

            # cheap histogram prune
            score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
            if score < hist_thresh:
                continue

            # BF match
            matches = sorted(estimator.bf.match(des1, des2),
                             key=lambda m: m.distance)
            if len(matches) < 4:
                continue

            keep = min(len(matches), max(4, int(len(matches)*0.8), 50))
            good = matches[:keep]

            # RANSAC
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            H_aff, mask = cv2.estimateAffinePartial2D(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=4.0,
                maxIters=5000,
                confidence=0.995
            )
            if H_aff is None or mask is None:
                continue

            inliers = int(mask.sum())
            print(f"[DEBUG] prev[{i}] vs curr[{j}]: inliers={inliers}")

            if inliers > best_inliers:
                best_inliers = inliers
                best_i, best_j = i, j
                best_H = np.vstack([H_aff, [0,0,1]]).astype(np.float64)

    if best_i < 0:
        raise RuntimeError("No reliable overlap found between those two clips.")
    print(f"[DEBUG] → Best overlap prev[{best_i}] ↔ curr[{best_j}] = {best_inliers} inliers")
    return best_i, best_j, best_H

def stitch(videos: List[str], stride: int, estimator_arg: str, out_path: str):
    # 1) Load each video into its own frame list
    sequences = [ sample_frames(v, stride) for v in videos ]
    if any(len(s)==0 for s in sequences):
        raise ValueError("One of the videos produced no frames!")

    # 2) Initialize
    canvas    = GrowableCanvas(sequences[0][0])
    estimator = MotionEstimator()
    H_cum     = np.eye(3, dtype=np.float64)
    prev      = sequences[0][0]

    # 3) Forward‑stitch the first clip in full
    for f in sequences[0][1:]:
        H_rel = estimator.feature_detection_and_matching(prev, f, -1, -1, videos[0])
        if H_rel is None:
            prev = f
            continue
        H_cum = np.linalg.inv(H_rel) @ H_cum
        canvas.add(f, H_cum)
        prev = f

    # 4) If there are more clips, process each one
    for vid_idx, seq in enumerate(sequences[1:], start=1):
        # <<< CHANGED: use full previous sequence for O(N²) search
        prev_seq = sequences[vid_idx-1]
        print(f"[Clip {vid_idx}] brute‑forcing pivot across {len(prev_seq)} prev‑frames × {len(seq)} new‑frames…")
        best_i, best_j, best_H = find_best_overlap(prev_seq, seq, estimator)

        cv2.imwrite("debug_prev_bi_vid_idx.png", prev_seq[best_i])
        cv2.imwrite("debug_curr_cj_vid_idx.png", seq[best_j])

        if best_i < 0:
            # No good overlap → just forward‑stitch this clip
            print(f"[Clip {vid_idx}] No overlap, falling back to forward‑only stitch")
            for f in seq:
                Hf = estimator.feature_detection_and_matching(prev, f, -1, -1, videos[vid_idx])
                if Hf is None:
                    prev = f
                    continue
                H_cum = np.linalg.inv(Hf) @ H_cum
                canvas.add(f, H_cum)
                prev = f
            continue  # onto the next clip

        # <<< CHANGED: pivot at seq[best_j], matched to prev_seq[best_i]
        print(f"[Clip {vid_idx}] Pivot: prev_seq[{best_i}] ↔ seq[{best_j}] (inliers in H)")
        H_cum = np.linalg.inv(best_H) @ H_cum
        canvas.add(seq[best_j], H_cum)
        prev = seq[best_j]

        # 4c) Back‑stitch everything *before* pivot
        H_back = H_cum.copy()
        for k in range(best_j-1, -1, -1):
            H_rb = estimator.feature_detection_and_matching(prev,
                                                            seq[k],
                                                            k+1,
                                                            k,
                                                            videos[vid_idx])
            if H_rb is None:
                prev = seq[k]
                continue
            H_back = H_back @ np.linalg.inv(H_rb)
            canvas.add(seq[k], H_back)
            prev = seq[k]

        # 4d) Forward‑stitch the rest *after* pivot
        prev = seq[best_j]
        H_fwd = H_cum.copy()
        for f in seq[best_j+1:]:
            H_rf = estimator.feature_detection_and_matching(prev,
                                                            f,
                                                            -1,
                                                            -1,
                                                            videos[vid_idx])
            if H_rf is None:
                prev = f
                continue
            H_fwd = np.linalg.inv(H_rf) @ H_fwd
            canvas.add(f, H_fwd)
            prev = f

    # 5) Render & save
    print("\nGenerating final canvas...")
    final = canvas.get_final_image()
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / out_path), final)
    print(f"Stitched panorama saved to {out_dir / out_path}")
