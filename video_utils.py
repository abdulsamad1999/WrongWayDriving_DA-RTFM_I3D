# video_utils.py

import torch
import numpy as np
import cv2
import os

def extract_sliding_clips(frames, clip_len=32, stride=16):
    clips = []
    indices = []
    for i in range(0, len(frames) - clip_len + 1, stride):
        clip = frames[i:i + clip_len]
        clips.append(torch.stack(clip, dim=0))  # (32, 3, 224, 224)
        indices.append((i, i + clip_len))
    return clips, indices

def annotate_anomalies_on_video(video_path, anomaly_intervals, output_path, fps=30):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    interval_idx = 0

    success, frame = cap.read()
    while success:
        is_anomalous = any(start <= frame_idx < end for (start, end) in anomaly_intervals)

        if is_anomalous:
            cv2.putText(frame, "⚠️ Anomaly Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 0, 255), 4)

        writer.write(frame)
        frame_idx += 1
        success, frame = cap.read()

    cap.release()
    writer.release()
