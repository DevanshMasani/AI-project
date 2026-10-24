"""
Train Neural A* (guidance-map encoder + differentiable A*) with Adam.

Example
-------
python make_synthetic_data.py --out-dir data
python train.py --train-data data/train.npz --val-data data/val.npz --epochs 100 --batch-size 32

Your own data: save an .npz with arrays
    maps   (N, 64, 64)  1 = passable, 0 = obstacle   (use --invert-map if it is the other way round)
    starts (N, 2)       (row, col)
    goals  (N, 2)       (row, col)
    paths  (N, 64, 64)  1 along the ground-truth path (incl. start and goal)
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neural_astar import NeuralAStar, PlanningDataset, evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Train Neural A* with Adam")
    p.add_argument("--train-data", required=True, help=".npz with maps/starts/goals/paths")
    p.add_argument("--val-data", required=True, help=".npz for validation (model selection)")
    p.add_argument("--invert-map", action="store_true", help="your maps use 1 = obstacle")
    p.add_argument("--encoder", default="vgg16", help="smp encoder name, e.g. vgg16 or vgg16_bn")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32, help="paper: 100; lower it if you run out of GPU memory")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=0.0, help="max grad norm, 0 = off")
    p.add_argument("--tau", type=float, default=None, help="temperature; default sqrt(map width)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", default="runs/neural_astar")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-vis", type=int, default=6, help="validation examples to plot at the end (0 = skip)")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_examples(model, dataset, device, path, n):
    """Plot: input | guidance map | search history + path | ground truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from neural_astar import plan_single

    n = min(n, len(dataset))
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.6 * n), squeeze=False)
    for i in range(n):
        sample = dataset[i]
        free = sample["map"][0].numpy()
        start = tuple(int(x) for x in dataset.starts[i])
        goal = tuple(int(x) for x in dataset.goals[i])
        phi, hist, path = plan_single(model, free, start, goal, device=device)

        def base_rgb():
            img = np.where(free[..., None] > 0.5, 0.8, 0.0) * np.ones((*free.shape, 3))
            return img

        inp = base_rgb()
        inp[start] = (0, 0.8, 0)
        inp[goal] = (1, 0, 0)

        res = base_rgb()
        res[hist > 0.5] = (0.2, 0.8, 0.2)
        res[path > 0.5] = (1, 0, 0)

        gt = base_rgb()
        gt[sample["path"][0].numpy() > 0.5] = (1, 0, 0)

        axes[i, 0].imshow(inp)
        axes[i, 1].imshow(phi, cmap="gray_r", vmin=0, vmax=1)
        axes[i, 2].imshow(res)
        axes[i, 3].imshow(gt)
        for j, t in enumerate(["input (S green, G red)", "guidance map (dark = costly)",
                               f"Neural A*: {int(hist.sum())} expanded", "ground truth"]):
            axes[i, j].set_title(t, fontsize=8)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    train_ds = PlanningDataset.from_npz(args.train_data, invert_map=args.invert_map)
    val_ds = PlanningDataset.from_npz(args.val_data, invert_map=args.invert_map)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    print(f"train samples: {len(train_ds)}   val samples: {len(val_ds)}   device: {device}")

    model = NeuralAStar(encoder_name=args.encoder, tau=args.tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params / 1e6:.2f}M")

    config = {"encoder_name": args.encoder, "tau": args.tau}
    log_path = os.path.join(args.out_dir, "log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_opt", "val_exp", "val_hmean", "seconds"])

    best_hmean = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            m, s, g, p = (batch[k].to(device) for k in ("map", "start", "goal", "path"))
            out, _ = model(m, s, g)
            # Eq. (7): mean L1 between the search history C and the ground-truth path map
            loss = F.l1_loss(out.histories, p)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            running += loss.item() * m.shape[0]
            seen += m.shape[0]
        train_loss = running / max(seen, 1)

        val = evaluate(model, val_loader, device)
        secs = time.time() - t0
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
              f"Opt {val['opt']:.1f} | Exp {val['exp']:.1f} | Hmean {val['hmean']:.1f} | {secs:.0f}s")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val["loss"], val["opt"], val["exp"], val["hmean"], secs])

        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "config": config, "val": val}
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))
        if val["hmean"] > best_hmean:          # model selection by Hmean, as in the paper
            best_hmean = val["hmean"]
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> new best Hmean {best_hmean:.1f}, saved best.pt")

    if args.n_vis > 0:
        try:
            best = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
            model.load_state_dict(best["model"])
            fig_path = os.path.join(args.out_dir, "examples.png")
            save_examples(model, val_ds, device, fig_path, args.n_vis)
            print(f"saved visualization to {fig_path}")
        except ImportError:
            print("matplotlib not installed, skipping visualization")


if __name__ == "__main__":
    main()
