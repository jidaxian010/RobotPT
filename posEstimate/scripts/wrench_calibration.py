"""
Wrench calibration: read a rosbag, plot Fx/Fy/Fz, and print per-axis averages.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rosbag_reader import AnyReader, typestore

TOPIC_NAME = "/rokubi/wrench"
BAG_PATH = Path("~/Downloads/bota_bag_test_wo_gripper").expanduser()


def get_stamp(msg, fallback_ns):
    try:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    except Exception:
        return fallback_ns * 1e-9


def read_wrench_force(bag_path, topic_name=TOPIC_NAME):
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
            samples.append((float(stamp), float(force.x), float(force.y), float(force.z)))

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
    padded = np.pad(series, half_window, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half_window + 1)
    local_med = np.median(windows, axis=1)
    local_mad = np.median(np.abs(windows - local_med[:, None]), axis=1)
    scale = np.maximum(1.4826 * local_mad, 1e-6)
    return np.abs(series - local_med) > k * scale


def remove_spikes(data, half_window=15, k=3.5, spike_pad=3, hard_limit=200.0):
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


def plot_force(data, title):
    t = data[:, 0]
    labels = ("Fx", "Fy", "Fz")
    colors = ("tab:red", "tab:green", "tab:blue")

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(t, data[:, idx + 1], color=colors[idx], linewidth=1.2, label=labels[idx])
        ax.set_ylabel("N")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Wrench calibration: plot forces and print per-axis averages.")
    parser.add_argument("bag_path", nargs="?", default=str(BAG_PATH), help="Path to rosbag")
    parser.add_argument("--topic", default=TOPIC_NAME, help="Wrench topic")
    args = parser.parse_args()

    raw_data = read_wrench_force(args.bag_path, topic_name=args.topic)
    clean_data = remove_spikes(raw_data)

    print(f"Loaded {len(raw_data)} samples, duration: {raw_data[-1, 0]:.3f} s")
    print(f"  Fx avg: {np.nanmean(clean_data[:, 1]):.4f} N")
    print(f"  Fy avg: {np.nanmean(clean_data[:, 2]):.4f} N")
    print(f"  Fz avg: {np.nanmean(clean_data[:, 3]):.4f} N")

    plot_force(clean_data, title=f"{Path(args.bag_path).name} {args.topic}")


if __name__ == "__main__":
    main()
