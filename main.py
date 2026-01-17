import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

import option

import random
import numpy as np

def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from config import Config
from dataset import Dataset
from model import Model, weight_init
from train import train

try:
    from utils import Visualizer
except Exception:
    Visualizer = None


def evaluate(test_loader, model, device):
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for feats, lbl in test_loader:
            feats = feats.to(device)  # (B,T,F)
            lbl = lbl.to(device)
            video_score, _, _ = model.infer(feats)
            all_scores.append(video_score.detach().cpu())
            all_labels.append(lbl.detach().cpu())

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    # AUC-ROC and AUC-PR (Average Precision)
    auc_roc = roc_auc_score(labels, scores) if len(set(labels.tolist())) > 1 else float('nan')
    auc_pr = average_precision_score(labels, scores) if len(set(labels.tolist())) > 1 else float('nan')
    return auc_roc, auc_pr


def main():
    args = option.parser.parse_args()

    # Reproducibility
    set_seed(int(getattr(args, 'seed', 123)), bool(getattr(args, 'deterministic', False)))

    # For GTA5 wrongway lists, splitting by label is required.
    if args.dataset.startswith('wrongway') and not args.split_by_label:
        args.split_by_label = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_feature_size = args.feature_size + (args.flow_dim if getattr(args, 'use_flow', False) else 0)

    # Datasets and loaders
    train_nset = Dataset(args, is_normal=True, test_mode=False)
    train_aset = Dataset(args, is_normal=False, test_mode=False)
    test_set = Dataset(args, is_normal=True, test_mode=True)

    train_nloader = DataLoader(train_nset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, drop_last=True)
    train_aloader = DataLoader(train_aset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers)

    # Model
    model = Model(n_features=input_feature_size, batch_size=args.batch_size).to(device)
    model.apply(weight_init)

    if args.pretrained_ckpt:
        ckpt = torch.load(args.pretrained_ckpt, map_location=device)
        model.load_state_dict(ckpt, strict=False)

    # Optimizer / LR schedule
    config = Config(args)
    optimizer = optim.Adam(model.parameters(), lr=config.lr[0], weight_decay=0.005)

    # Optional visualization
    viz = None
    if (not args.no_viz) and (Visualizer is not None):
        try:
            viz_env = args.viz_env if getattr(args, 'viz_env', None) else f'{args.dataset}-{args.model_name}'
            viz = Visualizer(env=viz_env, server=getattr(args, 'viz_server', 'http://localhost'), port=getattr(args, 'viz_port', 8097), use_incoming_socket=False)
        except Exception:
            viz = None

    base_out = getattr(args, 'output_dir', '.')
    ckpt_dir = os.path.join(base_out, 'ckpt')
    auc_dir = os.path.join(base_out, 'AUC-per-epoch')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(auc_dir, exist_ok=True)

    best_auc = -1.0
    print("\nStarting training...\n")

    for epoch in tqdm(range(1, args.max_epoch + 1), desc="Epoch"):
        # Update LR if schedule provided
        lr_now = config.lr[epoch - 1] if (epoch - 1) < len(config.lr) else config.lr[-1]
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        train(train_nloader, train_aloader, model, args.batch_size, optimizer, viz, device, args)

        # Evaluate
        auc_roc, auc_pr = evaluate(test_loader, model, device)
        print(f"Epoch {epoch:03d} | Test AUC-ROC: {auc_roc:.4f} | Test AUC-PR: {auc_pr:.4f}")

        if viz is not None:
            viz.plot_lines('Test AUC-ROC', auc_roc)
            viz.plot_lines('Test AUC-PR', auc_pr)


        # Save best
        if auc_roc > best_auc:
            best_auc = auc_roc
            ckpt_path = os.path.join(ckpt_dir, f'{args.model_name}_best.pkl')
            torch.save(model.state_dict(), ckpt_path)

            with open(os.path.join(auc_dir, 'best_auc.txt'), 'w', encoding='utf-8') as f:
                f.write(f"epoch={epoch}\nauc_roc={auc_roc}\nauc_pr={auc_pr}\n")

    # Save final
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{args.model_name}_final.pkl'))
    print("\nTraining complete. Best AUC-ROC:", best_auc)


if __name__ == '__main__':
    main()
