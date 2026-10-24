"""
Generate synthetic 64x64 path-planning data for train.py.

Maps (--style):
  noise       (default) every cell is an obstacle independently with probability ~density, giving the
              scattered single-pixel walls of a random-noise maze
  rectangles  random rectangles plus a few U-shaped dead ends
Ground truth: a shortest path found by BFS on the same 8-neighbour, unit-cost,
corner-cutting-allowed grid that the differentiable A* uses.

python make_synthetic_data.py --out-dir data --n-train 800 --n-val 100 --n-test 100
python make_synthetic_data.py --out-dir data --density 0.30 --density-jitter 0.05
"""
import argparse
import os
from collections import deque

import numpy as np

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def random_map(rng, size, n_rect, n_u):
    m = np.ones((size, size), dtype=np.uint8)           # 1 = free, 0 = wall
    for _ in range(n_rect):
        h = int(rng.integers(2, size // 4 + 1))
        w = int(rng.integers(2, size // 4 + 1))
        r = int(rng.integers(0, size - h + 1))
        c = int(rng.integers(0, size - w + 1))
        m[r:r + h, c:c + w] = 0
    for _ in range(n_u):                                  # U-shaped dead ends
        s = int(rng.integers(8, 17))
        ring = np.ones((s, s), dtype=bool)
        ring[2:s - 2, 2:s - 2] = False                    # hollow square, wall thickness 2
        ring[0:2, s // 2 - 2:s // 2 + 2] = False          # opening on the top side
        ring = np.rot90(ring, int(rng.integers(0, 4)))
        r = int(rng.integers(0, size - s + 1))
        c = int(rng.integers(0, size - s + 1))
        block = m[r:r + s, c:c + s]
        block[ring] = 0
    return m


def noise_map(rng, size, density, jitter):
    """Independent random obstacles; the density is jittered per map so maps differ in difficulty."""
    d = float(np.clip(density + rng.uniform(-jitter, jitter), 0.0, 0.6))
    return (rng.random((size, size)) >= d).astype(np.uint8)   # 1 = free, 0 = wall


def bfs_path(m, start, goal):
    """Shortest path (list of (r, c)) from start to goal, or None."""
    H, W = m.shape
    parent = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for dr, dc in NEIGHBORS:
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < H and 0 <= nc < W and m[nr, nc] == 1 and (nr, nc) not in parent:
                parent[(nr, nc)] = cur
                q.append((nr, nc))
    if goal not in parent:
        return None
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


def make_split(n, size, rng, min_dist, make_map):
    maps = np.zeros((n, size, size), dtype=np.uint8)
    paths = np.zeros((n, size, size), dtype=np.uint8)
    starts = np.zeros((n, 2), dtype=np.int64)
    goals = np.zeros((n, 2), dtype=np.int64)
    i = 0
    while i < n:
        m = make_map(rng)
        free = np.argwhere(m == 1)
        if len(free) < 2:
            continue
        for _ in range(50):                               # try several start/goal pairs on this map
            s = tuple(int(x) for x in free[rng.integers(len(free))])
            g = tuple(int(x) for x in free[rng.integers(len(free))])
            if max(abs(s[0] - g[0]), abs(s[1] - g[1])) < min_dist:
                continue
            path = bfs_path(m, s, g)
            if path is None:
                continue
            maps[i] = m
            starts[i], goals[i] = s, g
            for r, c in path:
                paths[i, r, c] = 1
            i += 1
            break
    return maps, starts, goals, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--min-dist", type=int, default=20, help="min Chebyshev distance between start and goal")
    ap.add_argument("--style", choices=["noise", "rectangles"], default="noise")
    ap.add_argument("--density", type=float, default=0.25, help="noise style: obstacle probability per cell")
    ap.add_argument("--density-jitter", type=float, default=0.05, help="noise style: +/- variation per map")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.style == "noise":
        make_map = lambda r: noise_map(r, args.size, args.density, args.density_jitter)
    else:
        make_map = lambda r: random_map(r, args.size, n_rect=int(r.integers(8, 20)), n_u=int(r.integers(1, 4)))
    for name, n in [("train", args.n_train), ("val", args.n_val), ("test", args.n_test)]:
        maps, starts, goals, paths = make_split(n, args.size, rng, args.min_dist, make_map)
        out = os.path.join(args.out_dir, f"{name}.npz")
        np.savez_compressed(out, maps=maps, starts=starts, goals=goals, paths=paths)
        print(f"{name}: {n} samples -> {out}  (mean path length {paths.sum((1, 2)).mean():.1f})")


if __name__ == "__main__":
    main()
