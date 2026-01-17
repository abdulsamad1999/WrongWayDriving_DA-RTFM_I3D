import os
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Extract optical-flow based motion features")
    parser.add_argument("--list", type=str, required=True,
                        help="Train/test list file (RGB feature path + label)")
    parser.add_argument("--video-root", type=str, required=True,
                        help="Root directory containing dataset videos")
    parser.add_argument("--suffix", type=str, default="_flow",
                        help="Suffix for saved flow feature files")
    parser.add_argument("--num-segments", type=int, default=32,
                        help="Number of temporal segments")
    parser.add_argument("--theta-ref-file", type=str, default="theta_ref.json",
                        help="JSON file storing dominant traffic direction (reuse across train/test)")
    return parser.parse_args()


def find_video(video_root, base_name):
    """Recursively find video whose filename contains base_name."""
    for root, _, files in os.walk(video_root):
        for f in files:
            if f.lower().endswith(".mp4") and base_name in f:
                return os.path.join(root, f)
    return None


def compute_optical_flow_features(video_path, num_segments):
    """
    Compute per-segment flow stats:
      returns array (num_segments, 4): [u_mean, v_mean, mag_mean, theta_mean]
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()

    if len(frames) < 2:
        return None

    total_frames = len(frames)
    seg_len = max(total_frames // num_segments, 1)

    flow_feats = []

    for s in range(num_segments):
        start = s * seg_len
        end = min(start + seg_len, total_frames - 1)

        u_vals, v_vals, mags, thetas = [], [], [], []

        for t in range(start, end):
            flow = cv2.calcOpticalFlowFarneback(
                frames[t], frames[t + 1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            u = flow[..., 0]
            v = flow[..., 1]

            mag = np.sqrt(u ** 2 + v ** 2)
            theta = np.arctan2(v, u)

            u_vals.append(float(np.mean(u)))
            v_vals.append(float(np.mean(v)))
            mags.append(float(np.mean(mag)))
            thetas.append(float(np.mean(theta)))

        if len(u_vals) == 0:
            flow_feats.append([0.0, 0.0, 0.0, 0.0])
        else:
            flow_feats.append([
                float(np.mean(u_vals)),
                float(np.mean(v_vals)),
                float(np.mean(mags)),
                float(np.mean(thetas))
            ])

    return np.array(flow_feats, dtype=np.float32)


def load_or_estimate_theta_ref(list_lines, video_root, num_segments, theta_ref_file):
    """
    If theta_ref_file exists -> load.
    Else -> estimate from NORMAL (label=0), non-flip videos in the provided list.
    """
    if os.path.exists(theta_ref_file):
        with open(theta_ref_file, "r") as f:
            theta_ref = float(json.load(f)["theta_ref"])
        print(f"[INFO] Loaded existing theta_ref from {theta_ref_file}: {theta_ref:.6f}")
        return theta_ref

    print(f"[INFO] theta_ref file not found: {theta_ref_file}")
    print("[INFO] Estimating theta_ref from NORMAL (label=0), non-flip videos in the provided list...")

    theta_vals = []

    for line in tqdm(list_lines, desc="Estimating theta_ref"):
        feat_path, label = line.split()
        label = int(label)

        if label != 0:
            continue
        if "_flip_" in feat_path:
            continue

        base = os.path.splitext(os.path.basename(feat_path))[0]
        video_path = find_video(video_root, base)
        if video_path is None:
            continue

        flow = compute_optical_flow_features(video_path, num_segments)
        if flow is None:
            continue

        theta_vals.extend(flow[:, 3].tolist())

    if len(theta_vals) == 0:
        raise RuntimeError("No normal non-flip videos found to estimate theta_ref.")

    theta_ref = float(np.mean(theta_vals))
    with open(theta_ref_file, "w") as f:
        json.dump({"theta_ref": theta_ref}, f, indent=2)

    print(f"[INFO] Saved theta_ref to {theta_ref_file}: {theta_ref:.6f}")
    return theta_ref


def main():
    args = parse_args()

    print("==== Optical Flow Feature Extraction ====")
    print("List file:", args.list)
    print("Video root:", args.video_root)
    print("Segments:", args.num_segments)
    print("Theta ref file:", args.theta_ref_file)
    print("Output suffix:", args.suffix)

    with open(args.list, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Load training theta_ref if present; otherwise estimate (for train run).
    theta_ref = load_or_estimate_theta_ref(lines, args.video_root, args.num_segments, args.theta_ref_file)

    # Extract features for all videos in list
    print("\n[STEP] Extracting flow features...")
    saved, missing = 0, 0

    for line in tqdm(lines, desc="Extracting flow"):
        feat_path, label = line.split()
        base = os.path.splitext(os.path.basename(feat_path))[0]

        video_path = find_video(args.video_root, base)
        if video_path is None:
            missing += 1
            print(f"[WARNING] Video not found for base name: {base}")
            continue

        flow = compute_optical_flow_features(video_path, args.num_segments)
        if flow is None:
            print(f"[WARNING] Flow failed (too short / unreadable) for: {video_path}")
            continue

        # Use theta_ref adjusted for flips
        if "_flip_" in base:
            theta_ref_use = np.pi - theta_ref
        else:
            theta_ref_use = theta_ref

        # Direction consistency
        D = np.cos(flow[:, 3] - theta_ref_use)

        # Final flow feature: [u, v, mag, D]
        flow_feat = np.stack([flow[:, 0], flow[:, 1], flow[:, 2], D], axis=1).astype(np.float32)

        out_path = os.path.splitext(feat_path)[0] + args.suffix + ".npy"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        np.save(out_path, flow_feat)
        saved += 1

    print(f"\n[✓] Done. Saved: {saved} flow feature files. Missing videos: {missing}.")
    print("Tip: For test extraction, always point --theta-ref-file to the TRAIN theta_ref.json.")


if __name__ == "__main__":
    main()
