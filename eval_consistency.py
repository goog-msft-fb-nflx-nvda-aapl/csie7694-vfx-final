"""
Temporal Edit-Consistency Evaluator for VideoCoF outputs.

Metrics:
  - flicker:  mean per-frame L2 diff in the edited-region mask across time
  - drift:    mean CLIP cosine similarity of edited frames vs frame-0 (identity preservation)
  - smoothness: mean optical-flow magnitude in edited region (lower = smoother)

Usage:
  python eval_consistency.py --edited results/obj_rem_1/gen_two_man.mp4 \
                             --source  assets/two_man.mp4 \
                             --tag     obj_rem_33

  python eval_consistency.py --edited results/longvid_128/gen_two_man.mp4 \
                             --source  assets/two_man.mp4 \
                             --tag     obj_rem_128
"""

import argparse
import json
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def read_video(path: str) -> np.ndarray:
    """Return (T, H, W, 3) uint8 array."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def diff_mask(edited: np.ndarray, source: np.ndarray, threshold: float = 20.0) -> np.ndarray:
    """
    Compute per-frame binary mask of edited region by comparing source vs edited.
    Returns (T, H, W) bool array. Frames beyond source length use last source frame.
    """
    T = edited.shape[0]
    S = source.shape[0]
    masks = []
    for t in range(T):
        s_frame = source[min(t, S - 1)].astype(float)
        e_frame = edited[t].astype(float)
        diff = np.linalg.norm(e_frame - s_frame, axis=-1)
        masks.append(diff > threshold)
    return np.stack(masks)


def compute_flicker(edited: np.ndarray, masks: np.ndarray) -> float:
    """Mean L2 diff between consecutive frames in edited region."""
    diffs = []
    for t in range(1, len(edited)):
        mask = masks[t] | masks[t - 1]
        if mask.sum() == 0:
            continue
        prev = edited[t - 1].astype(float)[mask]
        curr = edited[t].astype(float)[mask]
        diffs.append(np.mean(np.linalg.norm(curr - prev, axis=-1)))
    return float(np.mean(diffs)) if diffs else 0.0


def compute_flow_smoothness(edited: np.ndarray, masks: np.ndarray) -> float:
    """Mean optical-flow magnitude in edited region (lower = smoother)."""
    magnitudes = []
    for t in range(1, len(edited)):
        prev_gray = cv2.cvtColor(edited[t - 1], cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(edited[t], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag = np.linalg.norm(flow, axis=-1)
        mask = masks[t]
        if mask.sum() > 0:
            magnitudes.append(float(mag[mask].mean()))
    return float(np.mean(magnitudes)) if magnitudes else 0.0


def compute_clip_drift(edited: np.ndarray, masks: np.ndarray, device: str = "cuda") -> float:
    """
    Mean CLIP cosine similarity of each edited-region crop vs frame-0 crop.
    Lower drift = better temporal identity preservation.
    Returns 1 - mean_similarity (so lower is better, like flicker).
    """
    try:
        import clip
    except ImportError:
        print("clip not installed, skipping drift metric (pip install git+https://github.com/openai/CLIP.git)")
        return -1.0

    model, preprocess = clip.load("ViT-B/32", device=device)
    from PIL import Image

    def get_crop(frame, mask):
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return frame
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        return frame[y0:y1 + 1, x0:x1 + 1]

    ref_crop = get_crop(edited[0], masks[0] if masks[0].sum() > 0 else np.ones(masks[0].shape, bool))
    ref_img = preprocess(Image.fromarray(ref_crop)).unsqueeze(0).to(device)
    with torch.no_grad():
        ref_feat = model.encode_image(ref_img)
        ref_feat = F.normalize(ref_feat, dim=-1)

    sims = []
    for t in range(1, len(edited)):
        mask = masks[t] if masks[t].sum() > 0 else masks[0]
        crop = get_crop(edited[t], mask)
        img = preprocess(Image.fromarray(crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = F.normalize(model.encode_image(img), dim=-1)
        sims.append(float((ref_feat * feat).sum().cpu()))

    mean_sim = float(np.mean(sims)) if sims else 1.0
    return 1.0 - mean_sim  # drift: lower is better


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edited", required=True, help="Path to edited video")
    parser.add_argument("--source", required=True, help="Path to source video")
    parser.add_argument("--tag", default="", help="Label for this run")
    parser.add_argument("--out", default="eval_results.jsonl", help="Output JSONL file")
    parser.add_argument("--diff_threshold", type=float, default=20.0)
    parser.add_argument("--no_clip", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading videos...")
    edited = read_video(args.edited)
    source = read_video(args.source)
    T = len(edited)
    print(f"  edited: {T} frames, source: {len(source)} frames")

    print("Computing edit mask...")
    masks = diff_mask(edited, source, threshold=args.diff_threshold)
    edit_coverage = float(masks.mean())

    print("Computing flicker...")
    flicker = compute_flicker(edited, masks)

    print("Computing flow smoothness...")
    flow_mag = compute_flow_smoothness(edited, masks)

    drift = -1.0
    if not args.no_clip:
        print("Computing CLIP drift...")
        drift = compute_clip_drift(edited, masks, device=device)

    result = {
        "tag": args.tag,
        "edited": args.edited,
        "source": args.source,
        "num_frames": T,
        "edit_coverage": round(edit_coverage, 4),
        "flicker": round(flicker, 4),
        "flow_smoothness": round(flow_mag, 4),
        "clip_drift": round(drift, 4) if drift >= 0 else None,
    }

    print("\n=== Results ===")
    for k, v in result.items():
        print(f"  {k}: {v}")

    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"\nAppended to {args.out}")


if __name__ == "__main__":
    main()
