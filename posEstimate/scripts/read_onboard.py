import argparse
import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Allow imports from posEstimate/ regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rosbag_reader import AnyReader, typestore


TOPIC_NAME = "/rokubi/wrench"
DEFAULT_DATA_NAME = "yihenga2_onboard"
DEFAULT_BAG_PATH = Path(f"/home/jdx/Downloads/{DEFAULT_DATA_NAME}")
OUT_DIR = Path("posEstimate/data")


def get_stamp(msg, fallback_ns):
    try:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    except Exception:
        return fallback_ns * 1e-9


def read_wrench_force(bag_path, topic_name=TOPIC_NAME):
    """Read force xyz from a WrenchStamped topic."""
    bag_path = Path(bag_path)
    samples = []

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connection = next((c for c in reader.connections if c.topic == topic_name), None)
        if connection is None:
            raise RuntimeError(f"Topic {topic_name} not found in bag: {bag_path}")

        for conn, ts, raw in reader.messages(connections=[connection]):
            msg = typestore.deserialize_cdr(raw, conn.msgtype)
            stamp = get_stamp(msg, ts)
            force = msg.wrench.force
            samples.append(
                (
                    float(stamp),
                    float(force.x),
                    float(force.y),
                    float(force.z),
                )
            )

    if not samples:
        raise RuntimeError(f"No messages found on {topic_name} in bag: {bag_path}")

    data = np.asarray(samples, dtype=np.float64)
    data[:, 0] -= data[0, 0]
    return data


def _dilate_mask(mask, radius):
    if radius <= 0 or not np.any(mask):
        return mask
    expanded = mask.copy()
    for shift in range(1, radius + 1):
        expanded[:-shift] |= mask[shift:]
        expanded[shift:] |= mask[:-shift]
    return expanded


def _interp_masked(series, mask, fallback):
    if not np.any(mask):
        return series

    idx = np.arange(len(series))
    keep = ~mask
    if np.count_nonzero(keep) < 2:
        return fallback.copy()

    filled = series.copy()
    filled[mask] = np.interp(idx[mask], idx[keep], series[keep])
    return filled


def _hampel(series, half_window=15, k=2.0):
    """
    Hampel identifier: mark samples whose deviation from the local median
    exceeds k * (1.4826 * local MAD).  Uses a sliding window of size
    2*half_window+1, which keeps the baseline estimate purely local so that
    a cluster of spikes cannot inflate the threshold for the surrounding region.
    """
    n = len(series)
    padded = np.pad(series, half_window, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_window + 1)
    local_med = np.median(windows, axis=1)
    local_mad = np.median(np.abs(windows - local_med[:, None]), axis=1)
    scale = np.maximum(1.4826 * local_mad, 1e-6)
    return np.abs(series - local_med) > k * scale


def remove_spikes(
    data,
    half_window=15,
    k=3.5,
    spike_pad=3,
    hard_limit=200.0,
):
    """
    Remove sensor spikes from force data and restore continuity via linear
    interpolation.  Inlier samples are never modified.

    Single pass, two detectors:
      1. Hard limit  — flag any sample whose absolute value exceeds hard_limit.
      2. Hampel identifier — for each sample, compare it to the local median
         and local MAD inside a window of 2*half_window+1 samples.  Flag if
         deviation > k * (1.4826 * local_MAD).  The local window prevents a
         cluster of spikes from inflating the threshold for nearby inliers.
    Detected islands are expanded by spike_pad samples on each side to clip
    rising/falling edges, then gaps are filled by linear interpolation.

    NOTE: no derivative test is applied.  Real force transients (contact
    events) produce large legitimate derivatives and would be over-removed.
    """
    cleaned = data.copy()
    forces = cleaned[:, 1:]

    for axis in range(forces.shape[1]):
        series = forces[:, axis].copy()
        spikes_hard = np.abs(series) > hard_limit
        spikes_amp = _hampel(series, half_window=half_window, k=k)
        mask = _dilate_mask(spikes_hard | spikes_amp, spike_pad)
        forces[:, axis] = _interp_masked(series, mask, fallback=series)

    cleaned[:, 1:] = forces
    return cleaned


def plot_force(raw_data, smooth_data, title):
    t = smooth_data[:, 0]
    labels = ("Fx", "Fy", "Fz")
    colors = ("tab:red", "tab:green", "tab:blue")

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(t, smooth_data[:, idx + 1], color=colors[idx], linewidth=1.8, label=f"{labels[idx]} clean")
        ax.set_ylabel("N")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def save_force_rule_book(clean_data, bag_path, topic_name=TOPIC_NAME):
    """Save min/max force ranges per axis as a YAML rule book."""
    clean_data = np.asarray(clean_data, dtype=np.float64)
    force_axes = {
        "x": clean_data[:, 1],
        "y": clean_data[:, 2],
        "z": clean_data[:, 3],
    }

    axis_rules = {}
    for axis_name, values in force_axes.items():
        axis_rules[axis_name] = {
            "n_samples": int(len(values)),
            "min_N": round(float(np.min(values)), 2),
            "max_N": round(float(np.max(values)), 2),
            "mean_N": round(float(np.mean(values)), 2),
            "std_N": round(float(np.std(values)), 2),
        }

    bag_name = Path(bag_path).name
    doc = {
        "source_bag": bag_name,
        "topic": topic_name,
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": int(len(clean_data)),
        "force_ranges": axis_rules,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rules_path = OUT_DIR / f"{bag_name}_force_rules.yaml"
    with open(rules_path, "w") as f:
        f.write("# Force rule book\n")
        f.write("# Units are Newtons. Use min_N/max_N as per-axis safe-range limits.\n")
        f.write("# Load in Python: import yaml; rules = yaml.safe_load(open(path))\n\n")
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)

    print(f"Saved force rule book: {rules_path}")
    return rules_path


def main():
    parser = argparse.ArgumentParser(
        description="Read /rokubi/wrench force.x/y/z from a rosbag, remove invalid spikes, and plot."
    )
    parser.add_argument(
        "bag_path",
        nargs="?",
        default=str(DEFAULT_BAG_PATH),
        help="Path to rosbag directory or file",
    )
    parser.add_argument(
        "--topic",
        default=TOPIC_NAME,
        help="Wrench topic to read",
    )
    parser.add_argument(
        "--half-window",
        type=int,
        default=15,
        help="Half-width of the Hampel sliding window (samples)",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=3.5,
        help="Hampel MAD multiplier for amplitude spike detection (lower = more aggressive)",
    )
    parser.add_argument(
        "--spike-pad",
        type=int,
        default=3,
        help="Expand each detected spike island by this many neighboring samples",
    )
    parser.add_argument(
        "--hard-limit",
        type=float,
        default=200.0,
        help="Immediately discard any sample whose absolute force exceeds this value (N)",
    )
    args = parser.parse_args()

    raw_data = read_wrench_force(args.bag_path, topic_name=args.topic)
    smooth_data = remove_spikes(
        raw_data,
        half_window=args.half_window,
        k=args.k,
        spike_pad=args.spike_pad,
        hard_limit=args.hard_limit,
    )

    print(f"Loaded {len(raw_data)} wrench samples from {args.topic}")
    print(f"Duration: {raw_data[-1, 0]:.3f} s")
    save_force_rule_book(smooth_data, args.bag_path, topic_name=args.topic)
    plot_force(raw_data, smooth_data, title=f"{Path(args.bag_path).name} {args.topic}")


if __name__ == "__main__":
    main()
