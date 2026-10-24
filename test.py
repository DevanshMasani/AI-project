"""
Evaluate a trained Neural A* checkpoint on a test split.

python test.py --checkpoint runs/neural_astar/best.pt --data data/test.npz
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from neural_astar import NeuralAStar, PlanningDataset, evaluate
from train import save_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help=".npz with maps/starts/goals/paths")
    ap.add_argument("--invert-map", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-vis", type=int, default=8)
    ap.add_argument("--out-dir", default=None, help="default: folder of the checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = NeuralAStar(encoder_name=cfg["encoder_name"], tau=cfg["tau"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt['epoch']})")

    ds = PlanningDataset.from_npz(args.data, invert_map=args.invert_map)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    res = evaluate(model, loader, device)
    print(f"test samples: {len(ds)}")
    print(f"Opt {res['opt']:.2f} | Exp {res['exp']:.2f} | Hmean {res['hmean']:.2f} | L1 loss {res['loss']:.4f}")

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    if args.n_vis > 0:
        fig_path = os.path.join(out_dir, "test_examples.png")
        save_examples(model, ds, device, fig_path, args.n_vis)
        print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
