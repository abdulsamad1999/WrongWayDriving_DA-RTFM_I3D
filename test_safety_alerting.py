import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

import option
from dataset import Dataset
from model import Model


def segment_persistence_alert(seg_scores: np.ndarray, thr: float, k: int, m: int) -> np.ndarray:
    """
    seg_scores: (N, T)
    Returns alerts (N,) where alert=1 if within any window of length m,
    at least k segment scores >= thr.
    """
    N, T = seg_scores.shape
    hits = (seg_scores >= thr).astype(np.int32)

    if m <= 1:
        return (hits.sum(axis=1) >= k).astype(np.int32)

    if T < m:
        return (hits.sum(axis=1) >= k).astype(np.int32)

    cs = np.cumsum(hits, axis=1)
    prev = np.concatenate([np.zeros((N, 1), dtype=np.int32), cs[:, :-m]], axis=1)
    win_sum = cs[:, m - 1:] - prev  # (N, T-m+1)
    return (win_sum >= k).any(axis=1).astype(np.int32)


def metrics(labels: np.ndarray, preds: np.ndarray):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)
    rep = classification_report(labels, preds, digits=4, zero_division=0)
    return acc, f1, prec, rec, cm, rep


def candidate_thresholds(values: np.ndarray, max_unique: int = 2000) -> np.ndarray:
    v = values.reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.array([0.5], dtype=np.float64)

    uniq = np.unique(v)
    if uniq.size <= max_unique:
        return uniq

    qs = np.linspace(0.0, 1.0, max_unique)
    thr = np.quantile(v, qs)
    thr = np.unique(thr)
    return thr.astype(np.float64)


def select_video_threshold_max_recall(labels, video_scores, precision_floor: float):
    thr_list = candidate_thresholds(video_scores)
    best = None
    best_thr = 0.5
    best_pack = None

    for thr in thr_list:
        preds = (video_scores >= float(thr)).astype(np.int32)
        acc, f1, prec, rec, cm, rep = metrics(labels, preds)
        if prec < precision_floor:
            continue
        cand = (rec, prec, f1, -float(thr))
        if best is None or cand > best:
            best = cand
            best_thr = float(thr)
            best_pack = {"Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec, "CM": cm, "Report": rep}

    return best_thr, best_pack


def select_persistence_threshold_max_recall(labels, seg_scores, k: int, m: int, precision_floor: float):
    thr_list = candidate_thresholds(seg_scores)
    best = None
    best_thr = None
    best_pack = None

    for thr in thr_list:
        preds = segment_persistence_alert(seg_scores, float(thr), k=k, m=m)
        acc, f1, prec, rec, cm, rep = metrics(labels, preds)
        if prec < precision_floor:
            continue
        cand = (rec, prec, f1, -float(thr))
        if best is None or cand > best:
            best = cand
            best_thr = float(thr)
            best_pack = {"Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec, "CM": cm, "Report": rep}

    return best_thr, best_pack


def auto_search_persistence(labels, seg_scores, precision_floor: float, m_grid, k_grid_mode: str):
    """
    Search over (k,m) and threshold to maximize recall subject to precision >= precision_floor.

    Tie-breakers (in order):
      1) higher recall
      2) higher precision
      3) higher F1
      4) smaller m (faster alerting)
      5) smaller k (less strict, faster)
      6) lower threshold
    """
    best = None
    best_sel = None

    for m in m_grid:
        if k_grid_mode == "strict":
            k_list = [m, max(1, m - 1)]
        elif k_grid_mode == "medium":
            k_list = [max(1, m - 1), max(1, m - 2)]
        else:
            # permissive
            k_list = [max(1, m - 2), max(1, m - 3)]

        for k in k_list:
            k = int(min(max(k, 1), m))
            thr, pack = select_persistence_threshold_max_recall(labels, seg_scores, k=k, m=m, precision_floor=precision_floor)
            if pack is None:
                continue

            cand = (
                pack["Recall"],
                pack["Precision"],
                pack["F1"],
                -m,        # prefer smaller m
                -k,        # prefer smaller k
                -thr       # prefer lower threshold
            )

            if best is None or cand > best:
                best = cand
                best_sel = {"k": k, "m": m, "thr": thr, "pack": pack}

    return best_sel


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_feature_size = args.feature_size + (args.flow_dim if getattr(args, 'use_flow', False) else 0)

    model = Model(n_features=input_feature_size, batch_size=args.batch_size).to(device)
    sd = torch.load(args.model_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    test_set = Dataset(args, is_normal=True, test_mode=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    all_video_scores, all_seg_scores, all_labels = [], [], []

    with torch.no_grad():
        for feats, lbl in test_loader:
            feats = feats.to(device)
            lbl = lbl.to(device)

            video_score, seg_scores, _ = model.infer(feats)  # (B,), (B,T)

            all_video_scores.append(video_score.detach().cpu().numpy())
            all_seg_scores.append(seg_scores.detach().cpu().numpy())
            all_labels.append(lbl.detach().cpu().numpy())

    video_scores = np.concatenate(all_video_scores, axis=0).reshape(-1).astype(np.float64)
    seg_scores = np.concatenate(all_seg_scores, axis=0).astype(np.float64)
    labels = np.concatenate(all_labels, axis=0).reshape(-1).astype(np.int32)

    # Threshold-independent video-level metrics
    auc_roc = roc_auc_score(labels, video_scores) if len(np.unique(labels)) > 1 else float('nan')
    auc_pr = average_precision_score(labels, video_scores) if len(np.unique(labels)) > 1 else float('nan')

    precision_floor = float(args.precision_floor)

    # Option A: segment persistence (auto-search k,m,thr)
    persist_sel = None
    if args.alert_rule in ("segment_persistence", "auto"):
        m_grid = [int(x) for x in args.m_grid.split(",") if x.strip()]
        persist_sel = auto_search_persistence(labels, seg_scores, precision_floor, m_grid=m_grid, k_grid_mode=args.k_grid_mode)

    # Option B: video score threshold (max recall @ precision floor)
    vid_thr, vid_pack = (None, None)
    if args.alert_rule in ("video_score", "auto"):
        vid_thr, vid_pack = select_video_threshold_max_recall(labels, video_scores, precision_floor)

    # Choose the best option for safety alerting (maximize recall subject to precision floor)
    chosen = None
    if persist_sel is not None:
        chosen = {"type": "segment_persistence", **persist_sel}
    if vid_pack is not None:
        vid_choice = {"type": "video_score", "thr": float(vid_thr), "pack": vid_pack}
        if chosen is None:
            chosen = vid_choice
        else:
            # Compare by recall, then precision, then F1, then prefer persistence (more explainable for alerts)
            a = chosen["pack"]
            b = vid_choice["pack"]
            cand_a = (a["Recall"], a["Precision"], a["F1"])
            cand_b = (b["Recall"], b["Precision"], b["F1"])
            if cand_b > cand_a:
                chosen = vid_choice

    # If nothing meets the precision floor, fall back to best-F1 on video scores (still report)
    fallback_note = None
    if chosen is None:
        fallback_note = f"No operating point achieved precision >= {precision_floor:.2f}. Falling back to best F1 on video_score."
        thr_list = candidate_thresholds(video_scores)
        best_f1 = -1.0
        best_thr = 0.5
        best_pack = None
        for thr in thr_list:
            preds = (video_scores >= float(thr)).astype(np.int32)
            acc, f1, prec, rec, cm, rep = metrics(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
                best_pack = {"Accuracy": acc, "F1": f1, "Precision": prec, "Recall": rec, "CM": cm, "Report": rep}
        chosen = {"type": "video_score", "thr": best_thr, "pack": best_pack}

    # Reference operating point for transparency: video threshold=0.50
    ref_preds = (video_scores >= 0.5).astype(np.int32)
    ref_acc, ref_f1, ref_prec, ref_rec, ref_cm, _ = metrics(labels, ref_preds)

    return {
        "AUC-ROC": auc_roc,
        "AUC-PR": auc_pr,
        "chosen": chosen,
        "fallback_note": fallback_note,
        "ref": {"thr": 0.5, "Accuracy": ref_acc, "F1": ref_f1, "Precision": ref_prec, "Recall": ref_rec, "CM": ref_cm},
    }


def main():
    # Extend args without editing option.py
    option.parser.add_argument("--precision-floor", type=float, default=0.85,
                               help="Minimum precision for safety alerting operating point.")
    option.parser.add_argument("--alert-rule", type=str, default="auto",
                               choices=["auto", "segment_persistence", "video_score"],
                               help="Alerting rule to use. 'auto' selects the best (highest recall) subject to the precision floor.")
    option.parser.add_argument("--m-grid", type=str, default="4,5,6,8",
                               help="Comma-separated window sizes m to search for persistence rule.")
    option.parser.add_argument("--k-grid-mode", type=str, default="strict",
                               choices=["strict", "medium", "permissive"],
                               help="Which k values to try for each m (strict tries k=m and k=m-1).")
    option.parser.add_argument("--model-path", type=str, default=None,
                               help="Path to a checkpoint .pkl. If not set, will use ckpt/{model_name}_best.pkl if present else _final.pkl.")
    args = option.parser.parse_args()

    if args.dataset.startswith("wrongway") and not getattr(args, "split_by_label", False):
        args.split_by_label = True

    if args.model_path is None:
        best_path = os.path.join('ckpt', f'{args.model_name}_best.pkl')
        final_path = os.path.join('ckpt', f'{args.model_name}_final.pkl')
        args.model_path = best_path if os.path.exists(best_path) else final_path

    res = evaluate(args)
    chosen = res["chosen"]
    pack = chosen["pack"]

    print("=== Safety Alerting Evaluation (Precision-floor policy) ===")
    print(f"AUC-ROC (video-level): {res['AUC-ROC']:.4f}")
    print(f"AUC-PR  (video-level): {res['AUC-PR']:.4f}")

    if res["fallback_note"]:
        print("\n[NOTE]", res["fallback_note"])

    if chosen["type"] == "segment_persistence":
        print(f"\nChosen alert rule: segment_persistence (k={chosen['k']} of m={chosen['m']}) on segment scores")
        print(f"Selected threshold: {chosen['thr']:.4f}")
    else:
        print("\nChosen alert rule: video_score >= threshold")
        print(f"Selected threshold: {chosen['thr']:.4f}")

    print(f"Precision:{pack['Precision']:.4f}  (floor: {args.precision_floor:.2f})")
    print(f"Recall:   {pack['Recall']:.4f}")
    print(f"F1:       {pack['F1']:.4f}")
    print(f"Accuracy: {pack['Accuracy']:.4f}")
    print("Confusion Matrix:\n", pack["CM"])
    print("\nClassification Report:\n", pack["Report"])

    ref = res["ref"]
    print("\n--- Reference (video_score) @ Threshold = 0.50 ---")
    print(f"Precision:{ref['Precision']:.4f}  Recall:{ref['Recall']:.4f}  F1:{ref['F1']:.4f}  Acc:{ref['Accuracy']:.4f}")
    print("Confusion Matrix:\n", ref["CM"])


if __name__ == "__main__":
    main()
