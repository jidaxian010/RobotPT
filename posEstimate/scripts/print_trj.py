#!/usr/bin/env python3
"""
Print and plot a .trj trajectory file with labeled joints.

Usage:
    python print_trj.py -f path/to/file.trj
    python print_trj.py -f path/to/file.trj --no-plot
    python print_trj.py -f path/to/file.trj --rows 20
"""

import argparse
import sys
from pathlib import Path


# Set this to use a named file without the -f flag (e.g. "easy" or "pose3")
# Leave empty to require -f from the command line
TRAJ_NAME = "pose4"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
JOINT_LABELS = ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]


def parse_trj(path: Path):
    with open(path) as f:
        lines = f.readlines()

    size = int(lines[0].split()[0])
    duration = float(lines[1].split()[0])
    hz = float(lines[2].split()[0])

    # Line 4 is either "1 [type]" or empty — skip it
    data_start = 4

    frames = []
    for line in lines[data_start:]:
        line = line.strip().rstrip(',').strip()
        if not line:
            continue
        frames.append([float(v) for v in line.split()])

    return {"size": size, "duration": duration, "hz": hz, "frames": frames}


def print_header(trj, path: Path):
    print(f"\nFile   : {path}")
    print(f"Frames : {trj['size']}  (parsed: {len(trj['frames'])})")
    print(f"Duration: {trj['duration']:.6f} s")
    print(f"Rate   : {trj['hz']:.6f} Hz")
    n_joints = len(trj["frames"][0]) if trj["frames"] else 0
    labels = JOINT_LABELS[:n_joints]
    print(f"Joints : {n_joints}  ({', '.join(labels)})\n")


def print_rows(trj, n_rows: int):
    if not trj["frames"]:
        print("No data frames.")
        return

    n_joints = len(trj["frames"][0])
    labels = JOINT_LABELS[:n_joints]
    col_w = 12

    # Header row
    header = f"{'Frame':>6}  " + "  ".join(f"{lbl:>{col_w}}" for lbl in labels)
    print(header)
    print("-" * len(header))

    frames = trj["frames"]
    total = len(frames)
    rows_to_show = min(n_rows, total) if n_rows > 0 else total

    # If n_rows < total, show first half and last half
    if 0 < n_rows < total:
        half = n_rows // 2
        indices = list(range(half)) + ["..."] + list(range(total - half, total))
    else:
        indices = list(range(rows_to_show))

    for idx in indices:
        if idx == "...":
            print(f"{'...':>6}  " + "  ".join(f"{'...':>{col_w}}" for _ in labels))
            continue
        row = frames[idx]
        vals = "  ".join(f"{v:>{col_w}.6f}" for v in row)
        print(f"{idx + 1:>6}  {vals}")


def plot_trj(trj, path: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    frames = trj["frames"]
    if not frames:
        return

    n_joints = len(frames[0])
    labels = JOINT_LABELS[:n_joints]
    data = np.array(frames)
    t = np.arange(len(frames)) / trj["hz"]

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, lbl in enumerate(labels):
        ax.plot(t, data[:, i], linewidth=1.0, color=f"C{i}", label=lbl)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("joint position (rad)")
    ax.legend(loc="upper right", ncol=n_joints)
    ax.grid(True, linewidth=0.4)
    ax.set_title(path.name)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Print a .trj trajectory file with joint labels.")
    parser.add_argument("-f", "--file", default=None, help="Path to the .trj file (overrides TRAJ_NAME)")
    parser.add_argument("--rows", type=int, default=20,
                        help="Number of data rows to print (0 = all, default: 20)")
    parser.add_argument("--no-plot", action="store_true", help="Skip the matplotlib plot")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
    elif TRAJ_NAME:
        path = DATA_DIR / f"{TRAJ_NAME}.trj"
    else:
        print("Error: provide -f <file> or set TRAJ_NAME at the top of this script.", file=sys.stderr)
        sys.exit(1)

    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    trj = parse_trj(path)
    print_header(trj, path)
    print_rows(trj, args.rows)

    if not args.no_plot:
        plot_trj(trj, path)


if __name__ == "__main__":
    main()
