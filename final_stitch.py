#!/usr/bin/env python3
"""
Multi‑Video to Panorama stitcher – translation‑only with sliver concatenation

This pipeline:
 1. Samples frames from each input video.
 2. Stitches the first video sequentially.
 3. For every additional video:
    a) Finds the best overlapping frame across all previously stitched frames (or uses user-supplied pivots).
    b) Stitches frames before the pivot backwards.
    c) Stitches frames after the pivot forwards.
 4. Outputs one final panorama.

Usage
-----
python multi_stitch.py --videos clip1.mp4 clip2.mp4 clip3.mp4 --stride 2 --out panorama.png \
    --pivots 5:3 10:2

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
# Motion estimation – ORB + RANSAC affine
# ---------------------------------------------------------------------------
class MotionEstimator:
    def __init__(self, min_match_count: int = 10):
        self.orb = cv2.ORB_create(nfeatures=5000)
        self.bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_match_count = min_match_count

    def feature_detection_and_matching(self,
                                       frame1: np.ndarray,
                                       frame2: np.ndarray
                                       ) -> Optional[np.ndarray]:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        kp2, des2 = self.orb.detectAndCompute(gray2, None)
        if des1 is None or des2 is None:
            return None
        matches = self.bf.match(des1, des2)
        if len(matches) < self.min_match_count:
            return None
        matches = sorted(matches, key=lambda x: x.distance)[:min(50, len(matches))]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H_affine, mask = cv2.estimateAffinePartial2D(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=5000,
            confidence=0.995
        )
        if H_affine is None or mask is None or int(mask.sum()) < 4:
            return None
        return np.vstack([H_affine, [0,0,1]]).astype(np.float64)


# ---------------------------------------------------------------------------
# Canvas that grows to fit warped frames
# ---------------------------------------------------------------------------
class GrowableCanvas:
    def __init__(self):
        self.frames_data: List[Tuple[np.ndarray, np.ndarray]] = []
        self.initialized = False

    def _update_bounds(self, frame: np.ndarray, H_cum: np.ndarray):
        h, w = frame.shape[:2]
        corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32).reshape(-1,1,2)
        try:
            tc = cv2.perspectiveTransform(corners, H_cum)
        except cv2.error:
            tc = corners
        xs = tc[:,0,0]; ys = tc[:,0,1]
        mi, ma = xs.min(), xs.max()
        mj, mb = ys.min(), ys.max()
        if not self.initialized:
            self.min_x, self.max_x = mi, ma
            self.min_y, self.max_y = mj, mb
            self.initialized = True
        else:
            self.min_x, self.max_x = min(self.min_x,mi), max(self.max_x,ma)
            self.min_y, self.max_y = min(self.min_y,mj), max(self.max_y,mb)

    def add(self, frame: np.ndarray, H_cum: np.ndarray):
        if H_cum is None or H_cum.shape != (3,3):
            print("Warning: invalid H_cum, skipping frame.")
            return
        self._update_bounds(frame, H_cum)
        self.frames_data.append((frame.copy(), H_cum.copy()))

    def get_final_image(self) -> np.ndarray:
        if not self.frames_data:
            return np.zeros((100,100,3), dtype=np.uint8)
        fw = int(np.ceil(self.max_x - self.min_x))
        fh = int(np.ceil(self.max_y - self.min_y))
        fw, fh = max(fw,1), max(fh,1)
        T = np.array([[1,0,-self.min_x],[0,1,-self.min_y],[0,0,1]], dtype=np.float64)
        canvas = np.zeros((fh, fw, 3), dtype=np.uint8)
        print(f"Final canvas: {fw}×{fh}, x=[{self.min_x:.1f},{self.max_x:.1f}], y=[{self.min_y:.1f},{self.max_y:.1f}]")
        for i, (frm, H) in enumerate(self.frames_data):
            Hf = T @ H
            try:
                warped_frame = cv2.warpPerspective(frm, Hf, (fw,fh), flags=cv2.INTER_NEAREST)
            except cv2.error as e:
                print(f" Warp error on frame {i}: {e}")
                continue
            mask_new = (warped_frame.sum(axis=2)>0) & (canvas.sum(axis=2)==0)
            canvas[mask_new] = warped_frame[mask_new]
        return canvas


# ---------------------------------------------------------------------------
# Brute‑force pivot finder
# ---------------------------------------------------------------------------
def find_best_global_overlap(prev_frames: List[np.ndarray],
                             new_frames:  List[np.ndarray],
                             estimator:   MotionEstimator,
                             min_inliers: Optional[int] = None
                             ) -> Tuple[Optional[int], Optional[int], Optional[np.ndarray]]:
    best_i = best_j = None
    best_H = None
    best_in = 0
    threshold = min_inliers or estimator.min_match_count

    for i, pf in enumerate(prev_frames):
        gray1 = cv2.cvtColor(pf, cv2.COLOR_BGR2GRAY)
        kp1, des1 = estimator.orb.detectAndCompute(gray1, None)
        if des1 is None: continue
        for j, nf in enumerate(new_frames):
            gray2 = cv2.cvtColor(nf, cv2.COLOR_BGR2GRAY)
            kp2, des2 = estimator.orb.detectAndCompute(gray2, None)
            if des2 is None: continue
            matches = estimator.bf.match(des1, des2)
            if len(matches) < threshold: continue
            matches = sorted(matches, key=lambda x: x.distance)[:50]
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            H_aff, mask = cv2.estimateAffinePartial2D(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=4.0,
                maxIters=5000,
                confidence=0.995
            )
            if H_aff is None or mask is None: continue
            inliers = int(mask.sum())
            if inliers >= threshold and inliers > best_in:
                best_in, best_i, best_j = inliers, i, j
                best_H = np.vstack([H_aff, [0,0,1]]).astype(np.float64)
    return best_i, best_j, best_H


# ---------------------------------------------------------------------------
# Top‑level stitching
# ---------------------------------------------------------------------------
def stitch_videos(videos: List[str],
                  stride: int,
                  out_path: str,
                  pivots: Optional[List[Tuple[int,int]]] = None):
    sequences = [ sample_frames(v, stride) for v in videos ]
    if any(len(s)==0 for s in sequences):
        raise ValueError("One of the videos produced no frames!")

    canvas    = GrowableCanvas()
    estimator = MotionEstimator()

    # 1) Stitch the first video
    print(f"Stitching {videos[0]}...")
    H_cum = np.eye(3, dtype=np.float64)
    prev = sequences[0][0]
    canvas.add(prev, H_cum)
    for cur in sequences[0][1:]:
        Hrel = estimator.feature_detection_and_matching(prev, cur)
        if Hrel is None:
            print("  skip frame (no motion estimate)")
            prev = cur
            continue
        H_cum = np.linalg.inv(Hrel) @ H_cum
        H_cum /= H_cum[2,2]
        canvas.add(cur, H_cum)
        prev = cur

    # 2) Process each additional clip
    for k, vid in enumerate(videos[1:]):
        print(f"\nProcessing next video {vid}...")
        seq = sequences[k+1]

        # Gather stitched frames and their cumulative H
        prev_frames = [frm for frm,_ in canvas.frames_data]
        prev_Hs     = [H   for _,H   in canvas.frames_data]

        # Choose pivot
        if pivots and k < len(pivots):
            i_prev, j_pivot = pivots[k]
            print(f"  Using provided pivot: stitched frame {i_prev} ↔ new frame {j_pivot}")
            H_rel = estimator.feature_detection_and_matching(prev_frames[i_prev], seq[j_pivot])
            if H_rel is None:
                print("  Unable to compute H for provided pivot—skipping this clip.")
                continue
        else:
            i_prev, j_pivot, H_rel = find_best_global_overlap(prev_frames, seq, estimator)
        if i_prev is None or H_rel is None:
            print("  No sufficient overlap found—skipping this clip.")
            continue
        print(f"  Pivot: stitched frame {i_prev} ↔ new frame {j_pivot}")

        H_prev_cum = prev_Hs[i_prev]
        H_cum_piv  = np.linalg.inv(H_rel) @ H_prev_cum
        H_cum_piv /= H_cum_piv[2,2]
        canvas.add(seq[j_pivot], H_cum_piv)

        # Backward stitch
        H_back = H_cum_piv.copy()
        prev_b = seq[j_pivot]
        for m in range(j_pivot-1, -1, -1):
            Hrb = estimator.feature_detection_and_matching(prev_b, seq[m])
            if Hrb is None:
                prev_b = seq[m]
                continue
            H_back = H_back @ np.linalg.inv(Hrb)
            H_back /= H_back[2,2]
            canvas.add(seq[m], H_back)
            prev_b = seq[m]

        # Forward stitch
        H_fwd  = H_cum_piv.copy()
        prev_f = seq[j_pivot]
        for m in range(j_pivot+1, len(seq)):
            Hrf = estimator.feature_detection_and_matching(prev_f, seq[m])
            if Hrf is None:
                prev_f = seq[m]
                continue
            H_fwd = np.linalg.inv(Hrf) @ H_fwd
            H_fwd /= H_fwd[2,2]
            canvas.add(seq[m], H_fwd)
            prev_f = seq[m]

    # 3) Final panorama
    print("\nGenerating final panorama...")
    pano = canvas.get_final_image()
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / out_path), pano)
    print(f"Saved panorama to {out_dir / out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multi‑video panorama stitcher")
    ap.add_argument("--videos", nargs="+", required=True,
                    help="List of input video files (in stitch order)")
    ap.add_argument("--stride", "-s", type=int, default=1,
                    help="Frame sampling stride")
    ap.add_argument("--out", "-o", default="panorama.png",
                    help="Output panorama filename")
    ap.add_argument(
        "--pivots", "-p",
        nargs='+',
        default=[],
        help="Optional pivots for each extra clip, in prev_idx:new_idx format"
    )
    args = ap.parse_args()

    # parse pivot strings
    pivots: List[Tuple[int,int]] = []
    for pair in args.pivots:
        try:
            i, j = pair.split(":")
            pivots.append((int(i), int(j)))
        except:
            raise ValueError(f"Bad pivot '{pair}', must be prev_idx:new_idx")

    stitch_videos(args.videos, args.stride, args.out, pivots)
