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
    def __init__(self, min_match_count=10, use_homography=False):
        # Parameters for feature matching (ORB + RANSAC)
        self.orb = cv2.ORB_create(nfeatures=5000) # Use more features potentially
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.min_match_count = min_match_count
        self.use_homography = use_homography
        # No parameters needed for phase correlation itself

    def find_best_overlap(self, prev_frames: List[np.ndarray],
                          new_frames:  List[np.ndarray],
                          hist_thresh: float = 0.5,
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
            kp, des = self.orb.detectAndCompute(g, None)
            prev_feats.append((g, kp, des))

        curr_feats = []
        for f in new_frames:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(g, None)
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


        # --- Stochastic random sampling phase ---
        rng = np.random.default_rng()
        n_samples = min(200, n_prev * n_curr)
        sample_indices = set()
        while len(sample_indices) < n_samples:
            i = rng.integers(0, n_prev)
            j = rng.integers(0, n_curr)
            sample_indices.add((i, j))

        for idx, (i, j) in enumerate(sample_indices):
            g1, kp1, des1 = prev_feats[i]
            g2, kp2, des2 = curr_feats[j]
            if des1 is None or des2 is None:
                continue
            h1 = prev_hists[i]
            h2 = curr_hists[j]
            score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
            print(
                f"[INFO] Random overlap {idx+1}/{n_samples} | best so far: prev[{best_i}] curr[{best_j}] inliers={best_inliers}",
                end="\r"
            )
            if score < hist_thresh:
                continue
            matches = sorted(self.bf.match(des1, des2), key=lambda m: m.distance)
            if len(matches) < 4:
                continue
            keep = min(len(matches), max(4, int(len(matches)*0.8), 50))
            good = matches[:keep]
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            if self.use_homography:
                H, mask = cv2.findHomography(
                    pts1, pts2,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=4.0,
                    maxIters=5000,
                    confidence=0.995
                )
            else:
                H, mask = cv2.estimateAffinePartial2D(
                    pts1, pts2,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=4.0,
                    maxIters=5000,
                    confidence=0.995
                )
            if H is None or mask is None:
                continue
            inliers = int(mask.sum())
            if inliers > best_inliers:
                best_inliers = inliers
                best_i, best_j = i, j
                if self.use_homography:
                    best_H = H
                else:
                    best_H = np.vstack([H, [0,0,1]]).astype(np.float64)

        # --- Local refinement phase (window of 30 frames around best match) ---
        window = 20
        for i in range(max(0, best_i-window), min(n_prev, best_i+window+1)):
            for j in range(max(0, best_j-window), min(n_curr, best_j+window+1)):
                print(
                    f"[INFO] Local refine i={i} j={j} | best so far: prev[{best_i}] curr[{best_j}] inliers={best_inliers}",
                    end="\r"
                )
                g1, kp1, des1 = prev_feats[i]
                g2, kp2, des2 = curr_feats[j]
                if des1 is None or des2 is None:
                    continue
                h1 = prev_hists[i]
                h2 = curr_hists[j]
                score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
                if score < hist_thresh:
                    continue
                matches = sorted(self.bf.match(des1, des2), key=lambda m: m.distance)
                if len(matches) < 4:
                    continue
                keep = min(len(matches), max(4, int(len(matches)*0.8), 50))
                good = matches[:keep]
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                if self.use_homography:
                    H, mask = cv2.findHomography(
                        pts1, pts2,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=4.0,
                        maxIters=5000,
                        confidence=0.995
                    )
                else:
                    H, mask = cv2.estimateAffinePartial2D(
                        pts1, pts2,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=4.0,
                        maxIters=5000,
                        confidence=0.995
                    )
                if H is None or mask is None:
                    continue
                inliers = int(mask.sum())
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_i, best_j = i, j
                    if self.use_homography:
                        best_H = H
                    else:
                        best_H = np.vstack([H, [0,0,1]]).astype(np.float64)

        if best_i < 0:
            raise RuntimeError("No reliable overlap found between those two clips.")
        return best_i, best_j, best_H

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
        if hasattr(self, 'use_homography') and self.use_homography:
            # Use full projective homography
            H, mask = cv2.findHomography(
                points1, points2,
                method=cv2.RANSAC,
                ransacReprojThreshold=4.0,
                maxIters=5000,
                confidence=0.995
            )
            if H is None:
                print(f"Homography estimation failed between frames {frame_idx1}->{frame_idx2}")
                return None
            H_rel = H.astype(np.float64)
        else:
            # Use affine transform
            H_affine, mask = cv2.estimateAffinePartial2D(
                points1, points2,
                method=cv2.RANSAC,
                ransacReprojThreshold=4.0,
                maxIters=5000,
                confidence=0.995
            )
            if H_affine is None:
                print(f"Affine estimation failed between frames {frame_idx1}->{frame_idx2}")
                return None
            # promote 2×3 to 3×3 so the rest of the pipeline is unchanged
            H_rel = np.vstack([H_affine, [0, 0, 1]]).astype(np.float64)

        # Optional: Check the number of inliers
        if mask is not None:
            num_inliers = np.sum(mask)
            if num_inliers < 4:
                print(f"Transform found, but too few inliers ({num_inliers}) between frame {frame_idx1} and {frame_idx2}.")
                return None # Treat as failure
        else: # Should not happen if H_rel is not None, but check anyway
            print(f"Transform estimated but RANSAC mask is None between frame {frame_idx1} and {frame_idx2}.")
            return None

        return H_rel # Return the 3x3 homography or affine matrix


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


    def get_final_image(self, min_new_pixel_percent: float = 0.05) -> np.ndarray:
        """Create the final stitched panorama using perspective warping."""
        if not self.frames_data:
            raise ValueError("No frames to stitch!")

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
        last_status_length = 0
        for i, (frame, H_cum) in enumerate(self.frames_data): # Use H_cum
            # Combine H_cum with canvas translation T
            H_final = T @ H_cum # 3x3 multiplication

            try:
                # Use warpPerspective instead of warpAffine
                warped_frame = cv2.warpPerspective(frame, H_final, (final_w, final_h),
                                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

                # Alpha Blending (Averaging in overlap regions)
                # Convert canvas and warped_frame to float for blending calculations
                canvas_float = canvas.astype(np.float32)
                warped_frame_float = warped_frame.astype(np.float32)

                # Create alpha masks (H,W). A pixel has content if sum of its channels > 0.
                # Mask for existing content on the canvas (non-black pixels)
                alpha_canvas = (np.sum(canvas, axis=2) > 0).astype(np.float32)
                # Mask for new content from the warped frame (non-black pixels)
                alpha_warped = (np.sum(warped_frame, axis=2) > 0).astype(np.float32)

                # Only add frame if enough new pixels
                num_new_pixels = np.sum((alpha_warped == 1) & (alpha_canvas == 0))
                num_warped_pixels = np.sum(alpha_warped)
                percent_new = num_new_pixels / max(1, num_warped_pixels)
                if percent_new < min_new_pixel_percent:
                    status = (f"Skipping frame {i+1} ({percent_new*100:.2f}% new pixels) "
                            f"min_new_pixel_percent={min_new_pixel_percent*100:.2f}%, "
                            f"num_new_pixels={num_new_pixels}, num_warped_pixels={num_warped_pixels}")
                    print(status.ljust(last_status_length), end="\r")
                    last_status_length = len(status)
                    continue
                else:
                    status = f"Processing frame {i+1}/{len(self.frames_data)}"
                    print(status.ljust(last_status_length), end="\r")
                    last_status_length = len(status)

                # Expand mask dimensions to (H,W,1) for broadcasting with (H,W,3) images
                alpha_canvas_expanded = alpha_canvas[:, :, np.newaxis]
                alpha_warped_expanded = alpha_warped[:, :, np.newaxis]

                # Numerator for the blend: (canvas_content * its_weight) + (warped_content * its_weight)
                # Where a mask is 0, the corresponding term becomes 0.
                numerator = (canvas_float * alpha_canvas_expanded +
                             warped_frame_float * alpha_warped_expanded)

                # Denominator for the blend: sum of weights
                # This will be 0.0 where both are black, 1.0 where one has content, 2.0 where both have content.
                denominator = alpha_canvas_expanded + alpha_warped_expanded

                # Perform safe division. 
                # 'out=np.zeros_like(canvas_float)' initializes the output array with zeros.
                # 'where=denominator!=0' ensures division only happens where denominator is non-zero.
                # If denominator is 0 (both inputs black), the output pixel remains 0 from initialization.
                canvas_float = np.divide(numerator, denominator,
                                         out=np.zeros_like(canvas_float),
                                         where=denominator != 0)

                # Convert the blended result back to uint8 for the canvas
                canvas = canvas_float.astype(np.uint8)

            except cv2.error as e:
                print(f"Error warping frame {i} with warpPerspective: {e}")
                # Print relevant info for debugging
                h_f, w_f = frame.shape[:2]
                print(f"Frame shape: {h_f}x{w_f}, H_final: {H_final}")
                continue

        return canvas




def stitch(videos: List[str], stride: int, out_path: str, order: List[int], frames_per_debug: int = 20, min_new_pixel_percent: float = 0.05, use_homography: bool = False):

    # 1) Load each video into its own frame list
    sequences = [ sample_frames(v, stride) for v in videos ]

    debug_dir = Path("out/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    # Clear previous debug images if needed
    for item in debug_dir.glob('*.png'):
        item.unlink()

    # --- Initialize Canvas with the first frame ---
    canvas = GrowableCanvas()
    estimator = MotionEstimator(use_homography=use_homography)

    # --- Initialize Cumulative Transform ---
    # H_cum transforms points FROM frame 0's coord system TO frame i's coord system. Start with identity for i=0.
    H_cum = np.eye(3, dtype=np.float64) # Use 3x3 Identity for H_cum
    prev_frame = sequences[order[0]][0]
    canvas.add(prev_frame, H_cum) # Add first frame with identity H_cum

    # --- Maintain per-frame cumulative homographies for each segment ---
    # Assume: homographies[segment_idx][frame_idx] = cumulative H to origin
    homographies = []
    # For the first segment:
    first_frames = sequences[order[0]]
    H_cum = np.eye(3)
    prev_frame = first_frames[0]
    homographies_segment = [H_cum.copy()]
    for frame in first_frames[1:]:
        H_rel = estimator.feature_detection_and_matching(prev_frame, frame, 0, 0, videos[order[0]])
        if H_rel is not None:
            H_rel_inv = np.linalg.inv(H_rel)
            H_cum = H_rel_inv @ H_cum
            H_cum /= H_cum[2, 2]  # Normalize to keep numerical stability
            canvas.add(frame, H_cum)
            homographies_segment.append(H_cum.copy())
        if len(canvas.frames_data) % frames_per_debug == 0:
                print(f"Debug: Frame {len(canvas.frames_data)} |H_cum| = {np.linalg.det(H_cum):.4f}, bottom row = {H_cum[2]}")
                cv2.imwrite(str(debug_dir / f"debug_canvas_{len(canvas.frames_data):03d}.png"), canvas.get_final_image(min_new_pixel_percent))
        prev_frame = frame
    homographies.append(homographies_segment)

    # Process subsequent clips if any
    for clip_idx in range(1, len(order)):
        curr_frames = sequences[order[clip_idx]]
        prev_frames = sequences[order[clip_idx-1]]
        
        # Find best overlap between previous clip and current clip
        best_prev_idx, best_curr_idx, H_rel = estimator.find_best_overlap(prev_frames, curr_frames)

        # Debug: Save the best matching frames side by side
        best_prev_frame = prev_frames[best_prev_idx]
        best_curr_frame = curr_frames[best_curr_idx]
        match_vis = np.concatenate([
            cv2.resize(best_prev_frame, (best_prev_frame.shape[1], best_prev_frame.shape[0])),
            cv2.resize(best_curr_frame, (best_curr_frame.shape[1], best_curr_frame.shape[0]))
        ], axis=1)
        match_path = debug_dir / f"overlap_{clip_idx:03d}_best_match.png"
        cv2.imwrite(str(match_path), match_vis)

        # Get the cumulative H for the pivot frame in the previous segment
        H_prev_pivot = homographies[clip_idx-1][best_prev_idx]
        # Compute the cumulative H for the pivot frame in the current segment
        H_cum = H_prev_pivot @ np.linalg.inv(H_rel)
        H_cum /= H_cum[2, 2]
        
        # Store homographies for this segment
        homographies_segment = [None] * len(curr_frames)
        homographies_segment[best_curr_idx] = H_cum.copy()
        prev_frame = curr_frames[best_curr_idx]
        # Stitch forward from pivot
        for i in range(best_curr_idx, len(curr_frames)):
            frame = curr_frames[i]
            if i == best_curr_idx:
                canvas.add(frame, H_cum)
            else:
                H_rel = estimator.feature_detection_and_matching(prev_frame, frame, clip_idx, clip_idx, videos[order[clip_idx]])
                if H_rel is not None:
                    H_rel_inv = np.linalg.inv(H_rel)
                    H_cum = H_rel_inv @ H_cum
                    H_cum /= H_cum[2, 2]
                canvas.add(frame, H_cum)
                homographies_segment[i] = H_cum.copy()
                prev_frame = frame
            
            # Debug output
            if len(canvas.frames_data) % frames_per_debug == 0:
                print(f"Debug: Frame {len(canvas.frames_data)} |H_cum| = {np.linalg.det(H_cum):.4f}, bottom row = {H_cum[2]}")
                cv2.imwrite(str(debug_dir / f"debug_canvas_{len(canvas.frames_data):03d}.png"), canvas.get_final_image(min_new_pixel_percent))

        # Stitch backward from pivot
        H_cum_back = H_cum.copy()
        prev_frame = curr_frames[best_curr_idx]
        for i in range(best_curr_idx-1, -1, -1):
            frame = curr_frames[i]
            H_rel = estimator.feature_detection_and_matching(frame, prev_frame, clip_idx, clip_idx, videos[order[clip_idx]])
            if H_rel is not None:
                H_cum_back = H_rel @ H_cum_back
                H_cum_back /= H_cum_back[2, 2]
            canvas.add(frame, H_cum_back)
            homographies_segment[i] = H_cum_back.copy()
            prev_frame = frame

            # Debug output
            if len(canvas.frames_data) % frames_per_debug == 0:
                print(f"Debug: Frame {len(canvas.frames_data)} |H_cum| = {np.linalg.det(H_cum):.4f}, bottom row = {H_cum[2]}")
                cv2.imwrite(str(debug_dir / f"debug_canvas_{len(canvas.frames_data):03d}.png"), canvas.get_final_image(min_new_pixel_percent))

        homographies.append(homographies_segment)
        
    # --- Final Output ---
    print("\nGenerating final canvas...")
    final_image = canvas.get_final_image(min_new_pixel_percent=args.min_new_pixel_percent)
    out_dir = Path("out")
    cv2.imwrite(str(out_dir / out_path), final_image)
    print(f"Stitched panorama saved to {out_dir / out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Panorama stitcher – sliver-based')
    ap.add_argument('--videos', nargs='+', required=True, help='Directory or list of video files')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--order', type=int, nargs='+', help='Specify the order of video processing')
    ap.add_argument('--out', default='out.png', help='Output filename for the stitched panorama')
    ap.add_argument('--frames_per_debug', type=int, default=20)
    ap.add_argument('--homography', action='store_true', help='Use full projective homography instead of affine transform')
    ap.add_argument('--min_new_pixel_percent', type=float, default=0.05, help='Minimum percent of new pixels required to add a frame to the canvas (0.0-1.0)')
    args = ap.parse_args()

    # Support: if --videos is a single directory, expand to all .mp4 files in that directory
    if len(args.videos) == 1 and Path(args.videos[0]).is_dir():
        video_dir = Path(args.videos[0])
        args.videos = sorted(str(p) for p in video_dir.glob('*.mp4'))
        if not args.videos:
            ap.error(f"No .mp4 files found in directory: {video_dir}")
        print(f"Using all .mp4 files in {video_dir} as videos: {args.videos}")
    else:
        # Validate that all specified files exist
        for p in args.videos:
            if not Path(p).exists():
                ap.error(f"Video not found: {p}")

    # Default order if not specified
    if args.order is None:
        args.order = list(range(len(args.videos)))
    print("Creating output file: ", args.out)
    stitch(args.videos, args.stride, args.out, args.order, args.frames_per_debug, args.min_new_pixel_percent, args.homography)