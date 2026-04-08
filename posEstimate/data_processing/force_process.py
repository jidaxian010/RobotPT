"""
ForceProcessor — loads, calibrates, smooths, and plots the full 6-DOF wrench
(Fx/Fy/Fz + Tx/Ty/Tz) from a MoCap rosbag using the /rokubi/wrench topic.
"""

import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml

from data_processing.rosbag_reader import AnyReader, typestore

FORCE_TOPIC = "/rokubi/wrench"

# Sensor → gripper: +x_s → -x_g,  +y_s → -y_g,  +z_s → +z_g
R_SENSOR2GRIPPER = np.array([
    [-1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
], dtype=np.float64)

# Gripper → EE: +x_g → -y_ee,  +y_g → -z_ee,  +z_g → +x_ee
R_GRIPPER2EE = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)

R_SENSOR2EE = R_GRIPPER2EE @ R_SENSOR2GRIPPER


def _get_stamp(msg, fallback_ns):
    try:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    except Exception:
        return fallback_ns * 1e-9


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
    channels = cleaned[:, 1:]
    for axis in range(channels.shape[1]):
        series = channels[:, axis].copy()
        spikes_hard = np.abs(series) > hard_limit
        spikes_amp = _hampel(series, half_window=half_window, k=k)
        mask = _dilate_mask(spikes_hard | spikes_amp, spike_pad)
        channels[:, axis] = _interp_masked(series, mask, fallback=series)
    cleaned[:, 1:] = channels
    return cleaned


class ForceProcessor:
    def __init__(
        self,
        bag_path,
        output_dir=None,
        topic=FORCE_TOPIC,
        show_image=True,
        offset=None,
        smooth=True,
        half_window=15,
        k=3.5,
        spike_pad=3,
        hard_limit=200.0,
    ):
        self.bag_path = Path(bag_path)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.topic = topic
        self.show_image = bool(show_image)
        self.offset = np.asarray(offset, dtype=np.float64) if offset is not None else None
        self.smooth = smooth
        self.half_window = half_window
        self.k = k
        self.spike_pad = spike_pad
        self.hard_limit = hard_limit

        # columns: [t, Fx, Fy, Fz, Tx, Ty, Tz]
        self.data_raw = None
        self.data_smoothed = None   # smoothed, before offset subtraction
        self.data = None            # smoothed + calibrated, sensor frame
        self.data_ee = None         # self.data rotated into EE frame

    def load(self):
        samples = []

        with AnyReader([self.bag_path], default_typestore=typestore) as reader:
            conn = next(
                (c for c in reader.connections if c.topic == self.topic), None
            )
            if conn is None:
                raise RuntimeError(
                    f"Topic {self.topic!r} not found in {self.bag_path}.\n"
                    f"Available topics: {[c.topic for c in reader.connections]}"
                )

            for conn_, ts, raw in reader.messages(connections=[conn]):
                msg = typestore.deserialize_cdr(raw, conn_.msgtype)
                stamp = _get_stamp(msg, ts)
                f = msg.wrench.force
                t = msg.wrench.torque
                samples.append([
                    float(stamp),
                    float(f.x), float(f.y), float(f.z),
                    float(t.x), float(t.y), float(t.z),
                ])

        if not samples:
            raise RuntimeError(
                f"No messages on {self.topic!r} in {self.bag_path}"
            )

        self.data_raw = np.asarray(samples, dtype=np.float64)
        self.data_raw[:, 0] -= self.data_raw[0, 0]   # zero-base timestamps

        print(
            f"  Loaded {len(self.data_raw)} wrench samples, "
            f"duration: {self.data_raw[-1, 0]:.3f} s"
        )

    def smooth_data(self):
        if self.smooth:
            self.data = remove_spikes(
                self.data_raw,
                half_window=self.half_window,
                k=self.k,
                spike_pad=self.spike_pad,
                hard_limit=self.hard_limit,
            )
            print(
                f"  Applied spike removal: half_window={self.half_window}, "
                f"k={self.k}, spike_pad={self.spike_pad}, hard_limit={self.hard_limit}"
            )
        else:
            self.data = self.data_raw.copy()

        if self.offset is not None:
            self.data_smoothed = self.data.copy()      # smoothed, before offset
            self.data[:, 1:] -= self.offset            # subtract calibration offset
        else:
            self.data_smoothed = None

    def to_ee_frame(self):
        """Rotate calibrated wrench from sensor frame into EE frame."""
        self.data_ee = self.data.copy()
        self.data_ee[:, 1:4] = (R_SENSOR2EE @ self.data[:, 1:4].T).T   # force
        self.data_ee[:, 4:7] = (R_SENSOR2EE @ self.data[:, 4:7].T).T   # torque

    def _plot_wrench(self, data, title, suffix=""):
        t = data[:, 0]
        force_labels  = (f"Fx{suffix}", f"Fy{suffix}", f"Fz{suffix}")
        torque_labels = (f"Tx{suffix}", f"Ty{suffix}", f"Tz{suffix}")
        force_colors  = ("tab:red",    "tab:green", "tab:blue")
        torque_colors = ("tab:orange", "tab:cyan",  "tab:purple")

        fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
        for i in range(3):
            axes[i].plot(t, data[:, i + 1], color=force_colors[i],
                         linewidth=1.2, label=force_labels[i])
            axes[i].set_ylabel("N")
            axes[i].grid(True, linewidth=0.4, alpha=0.5)
            axes[i].legend(loc="upper right")
        for i in range(3):
            axes[i + 3].plot(t, data[:, i + 4], color=torque_colors[i],
                             linewidth=1.2, label=torque_labels[i])
            axes[i + 3].set_ylabel("Nm")
            axes[i + 3].grid(True, linewidth=0.4, alpha=0.5)
            axes[i + 3].legend(loc="upper right")
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(title)
        plt.tight_layout()

    def plot_results(self):
        if not self.show_image:
            print("Skipping wrench plot (show_image=False).")
            return
        if self.data is None:
            print("No wrench data to plot.")
            return

        base_title = f"{self.bag_path.stem}  {self.topic}"
        t = self.data[:, 0]
        force_colors  = ("tab:red",    "tab:green", "tab:blue")
        torque_colors = ("tab:orange", "tab:cyan",  "tab:purple")

        # sensor-frame plot with optional raw overlay
        fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
        labels_f = ("Fx", "Fy", "Fz")
        labels_t = ("Tx", "Ty", "Tz")
        for i in range(3):
            if self.data_smoothed is not None:
                axes[i].plot(t, self.data_smoothed[:, i + 1], color=force_colors[i],
                             linewidth=0.8, alpha=0.35, label=f"{labels_f[i]} raw")
            axes[i].plot(t, self.data[:, i + 1], color=force_colors[i],
                         linewidth=1.2, label=labels_f[i])
            axes[i].set_ylabel("N")
            axes[i].grid(True, linewidth=0.4, alpha=0.5)
            axes[i].legend(loc="upper right")
        for i in range(3):
            if self.data_smoothed is not None:
                axes[i + 3].plot(t, self.data_smoothed[:, i + 4], color=torque_colors[i],
                                 linewidth=0.8, alpha=0.35, label=f"{labels_t[i]} raw")
            axes[i + 3].plot(t, self.data[:, i + 4], color=torque_colors[i],
                             linewidth=1.2, label=labels_t[i])
            axes[i + 3].set_ylabel("Nm")
            axes[i + 3].grid(True, linewidth=0.4, alpha=0.5)
            axes[i + 3].legend(loc="upper right")
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{base_title}  [sensor frame]")
        plt.tight_layout()

    def plot_ee_frame(self):
        if not self.show_image:
            return
        if self.data_ee is None:
            print("No EE-frame data to plot.")
            return
        base_title = f"{self.bag_path.stem}  {self.topic}"
        self._plot_wrench(self.data_ee, title=f"{base_title}  [EE frame]")

    def save_force_rules(self):
        """Write min/max of the EE-frame wrench to force_rule.yaml."""
        if self.data_ee is None:
            print("No EE-frame data — skipping force_rule.yaml.")
            return None

        out_dir = self.output_dir if self.output_dir is not None else self.bag_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        ee = self.data_ee
        axes_force  = {"x": ee[:, 1], "y": ee[:, 2], "z": ee[:, 3]}
        axes_torque = {"x": ee[:, 4], "y": ee[:, 5], "z": ee[:, 6]}

        def _range(values):
            return {
                "min": round(float(np.nanmin(values)), 4),
                "max": round(float(np.nanmax(values)), 4),
            }

        doc = {
            "source_bag": self.bag_path.name,
            "topic": self.topic,
            "frame": "EE",
            "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": int(len(ee)),
            "force_N": {ax: _range(v) for ax, v in axes_force.items()},
            "torque_Nm": {ax: _range(v) for ax, v in axes_torque.items()},
        }

        yaml_path = out_dir / "force_rule.yaml"
        with open(yaml_path, "w") as f:
            f.write("# Wrench rule book — EE frame\n")
            f.write("# Load: import yaml; rules = yaml.safe_load(open(path))\n\n")
            yaml.dump(doc, f, default_flow_style=False, sort_keys=False)

        print(f"  Saved force rules: {yaml_path}")
        return yaml_path

    def run(self):
        self.load()
        self.smooth_data()
        self.to_ee_frame()
        self.save_force_rules()
        self.plot_results()
        self.plot_ee_frame()
