"""
Render one training example as a 64x64 RGB image.

    R channel = goal map      (1 at the goal cell)
    G channel = start map     (1 at the start cell)
    B channel = free map      (1 where there is NO obstacle)

Because the channels are independent, the start cell (free) shows up cyan (0,255,255) and the goal
cell magenta (255,0,255); free space is blue, walls are black. Use --pure to make start pure green
and goal pure red instead.

Usage
-----
python view_example.py --data data/train.npz --index 5
python view_example.py --data data/train.npz --random
python view_example.py --data data/train.npz --index 5 --pure --out ex5.png

Outputs <out> (exact 64x64) and <out stem>_x<scale>.png (enlarged with nearest-neighbour for viewing).
"""
import argparse
import os

import numpy as np


def build_rgb(free, start_rc, goal_rc, pure=False):
    """free: (H, W) bool/0-1, start_rc / goal_rc: (row, col). Returns (H, W, 3) uint8."""
    H, W = free.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 2] = (free > 0.5).astype(np.uint8) * 255      # B: free space
    rgb[start_rc[0], start_rc[1], 1] = 255                 # G: start
    rgb[goal_rc[0], goal_rc[1], 0] = 255                   # R: goal
    if pure:                                               # start -> pure green, goal -> pure red
        rgb[start_rc[0], start_rc[1], 2] = 0
        rgb[goal_rc[0], goal_rc[1], 2] = 0
    return rgb


def render_pair(free, start_rc, goal_rc, phi, out_path, pure=False, auto_range=False,
                mask_walls=False, title=""):
    """Left: the map in the R=goal / G=start / B=free encoding. Right: the guidance map phi in ONE colour
    (white = 0, dark orange = high), with a colour bar. phi has the same HxW shape as the map.

    auto_range : scale the colours to [min(phi), max(phi)] instead of the fixed [0, 1]. Useful when phi
                 varies only a little, since a fixed range then looks almost uniform.
    mask_walls : paint obstacle cells black in the guidance panel (phi there is never used by the search).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    free = np.asarray(free) > 0.5
    rgb = build_rgb(free, start_rc, goal_rc, pure=pure)
    p = np.asarray(phi, dtype=float)
    cmap = plt.get_cmap("Oranges").copy()
    if mask_walls:
        p = np.ma.masked_where(~free, p)
        cmap.set_bad("black")
    vmin, vmax = (float(p.min()), float(p.max())) if auto_range else (0.0, 1.0)
    if vmax - vmin < 1e-9:
        vmax = vmin + 1e-9

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.4))
    ax[0].imshow(rgb, interpolation="nearest")
    ax[0].set_title("map  (R = goal, G = start, B = free)" + (f"\n{title}" if title else ""), fontsize=10)
    im = ax[1].imshow(p, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax[1].set_title("guidance map (darker = higher cost)", fontsize=10)
    cbar = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.set_label("guidance cost")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_png(arr, path):
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
    except ImportError:
        import matplotlib.pyplot as plt
        plt.imsave(path, arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help=".npz with maps/starts/goals/paths")
    ap.add_argument("--index", type=int, default=0, help="which example to render")
    ap.add_argument("--random", action="store_true", help="pick a random example instead of --index")
    ap.add_argument("--invert-map", action="store_true", help="your maps use 1 = obstacle")
    ap.add_argument("--pure", action="store_true", help="start pure green, goal pure red")
    ap.add_argument("--scale", type=int, default=10, help="enlargement factor for the preview image")
    ap.add_argument("--out", default="example.png", help="path of the exact 64x64 image")
    args = ap.parse_args()

    d = np.load(args.data)
    maps, starts, goals = d["maps"], d["starts"], d["goals"]
    n = maps.shape[0]
    idx = int(np.random.randint(n)) if args.random else args.index
    if not 0 <= idx < n:
        raise SystemExit(f"--index must be in [0, {n - 1}], got {idx}")

    free = maps[idx] > 0.5
    if args.invert_map:
        free = ~free
    start, goal = tuple(int(x) for x in starts[idx]), tuple(int(x) for x in goals[idx])
    rgb = build_rgb(free, start, goal, pure=args.pure)

    stem, ext = os.path.splitext(args.out)
    big_path = f"{stem}_x{args.scale}{ext or '.png'}"
    save_png(rgb, args.out)
    save_png(np.kron(rgb, np.ones((args.scale, args.scale, 1), dtype=np.uint8)), big_path)

    path_len = int((d["paths"][idx] > 0.5).sum()) if "paths" in d.files else None
    print(f"example {idx} of {n}")
    print(f"  map size   : {free.shape[0]}x{free.shape[1]}, free cells {free.mean() * 100:.1f}%")
    print(f"  start (row, col): {start}   goal (row, col): {goal}")
    if path_len is not None:
        print(f"  ground-truth path length: {path_len} nodes")
    print(f"saved {args.out} (64x64 exact) and {big_path} (enlarged)")


if __name__ == "__main__":
    main()
