"""
Process Calibration_Bag_2: gripper-only attached to the force sensor.

Pipeline:
  1. Load right-camera IMU → rotate to gripper frame → rotate to sensor frame.
  2. Load wrench → subtract sensor bias (from bota_bag_test_wo_gripper).
  3. Synchronise IMU and wrench to a common timeline via interpolation.
  4. Fit GripperCompensator (mass + CoM) via least squares.
  5. Save compensator to gripper_comp.npz next to the bag.
  6. Optionally plot diagnostics.
"""

import sys
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_processing.rosbag_reader import AnyReader, typestore, RosbagReader
from data_processing.wrench_calibration import WrenchCalibrator
from data_processing.force_process import remove_spikes
from data_processing.gripper_compensation import GripperCompensator
from imu.object_frame import ObjectFrame

# ==========================================
# --- 1. USER CONFIGURATION ---
# ==========================================

DATA_NAME      = "Calibration_Bag_2"
BAG_PATH       = Path(f"~/Downloads/{DATA_NAME}").expanduser()
CALIB_BAG_PATH = Path("~/Downloads/bota_bag_test_wo_gripper").expanduser()

FORCE_TOPIC   = "/rokubi/wrench"
IMU_TOPIC     = "/right_camera/camera/camera/imu"
GRIPPER_MASS  = 1.543   # kg

# Sensor → gripper: +x_s → -x_g,  +y_s → -y_g,  +z_s → +z_g
R_SENSOR2GRIPPER = np.array([
    [-1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
], dtype=np.float64)

SHOW_IMAGE = True


# ==========================================
# --- 2. HELPERS ---
# ==========================================

def _get_stamp(msg, fallback_ns):
    try:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    except Exception:
        return fallback_ns * 1e-9


def load_imu_right(bag_path):
    """
    Read right-camera IMU from bag.
    Returns np.ndarray shape (N, 7): [t, ax, ay, az, gx, gy, gz]
    """
    samples = []
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conn = next((c for c in reader.connections if c.topic == IMU_TOPIC), None)
        if conn is None:
            raise RuntimeError(
                f"Topic {IMU_TOPIC!r} not found in {bag_path}.\n"
                f"Available: {[c.topic for c in reader.connections]}"
            )
        for conn_, ts, raw in reader.messages(connections=[conn]):
            msg = typestore.deserialize_cdr(raw, conn_.msgtype)
            t = _get_stamp(msg, ts)
            a = msg.linear_acceleration
            g = msg.angular_velocity
            samples.append([t, float(a.x), float(a.y), float(a.z),
                               float(g.x), float(g.y), float(g.z)])

    if not samples:
        raise RuntimeError(f"No messages on {IMU_TOPIC!r} in {bag_path}")

    data = np.asarray(samples, dtype=np.float64)
    data[:, 0] -= data[0, 0]
    print(f"  Loaded {len(data)} IMU samples, duration: {data[-1, 0]:.3f} s")
    return data


def load_force(bag_path):
    """
    Read 6-DOF wrench from bag.
    Returns np.ndarray shape (N, 7): [t, Fx, Fy, Fz, Tx, Ty, Tz]
    """
    samples = []
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conn = next((c for c in reader.connections if c.topic == FORCE_TOPIC), None)
        if conn is None:
            raise RuntimeError(
                f"Topic {FORCE_TOPIC!r} not found in {bag_path}.\n"
                f"Available: {[c.topic for c in reader.connections]}"
            )
        for conn_, ts, raw in reader.messages(connections=[conn]):
            msg = typestore.deserialize_cdr(raw, conn_.msgtype)
            t = _get_stamp(msg, ts)
            f = msg.wrench.force
            tor = msg.wrench.torque
            samples.append([t, float(f.x), float(f.y), float(f.z),
                               float(tor.x), float(tor.y), float(tor.z)])

    if not samples:
        raise RuntimeError(f"No messages on {FORCE_TOPIC!r} in {bag_path}")

    data = np.asarray(samples, dtype=np.float64)
    data[:, 0] -= data[0, 0]
    print(f"  Loaded {len(data)} wrench samples, duration: {data[-1, 0]:.3f} s")
    return data


# ==========================================
# --- 3. PLOTTING ---
# ==========================================

def plot_imu_gripper(imu_raw, imu_gripper):
    """Plot right-IMU accel and gyro in gripper frame."""
    t = imu_raw[:, 0]
    fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    fig.suptitle("Right IMU → Gripper Frame")

    accel_labels  = ("ax", "ay", "az")
    gyro_labels   = ("gx", "gy", "gz")
    accel_colors  = ("tab:red", "tab:green", "tab:blue")
    gyro_colors   = ("tab:orange", "tab:cyan", "tab:purple")

    for i in range(3):
        axes[i].plot(t, imu_gripper[:, i + 1], color=accel_colors[i],
                     linewidth=1.2, label=f"{accel_labels[i]} (gripper)")
        axes[i].plot(t, imu_raw[:, i + 1], color=accel_colors[i],
                     linewidth=0.7, alpha=0.3, label=f"{accel_labels[i]} (raw)")
        axes[i].set_ylabel("m/s²")
        axes[i].legend(loc="upper right")
        axes[i].grid(True, linewidth=0.4, alpha=0.5)

    for i in range(3):
        axes[i + 3].plot(t, imu_gripper[:, i + 4], color=gyro_colors[i],
                         linewidth=1.2, label=f"{gyro_labels[i]} (gripper)")
        axes[i + 3].plot(t, imu_raw[:, i + 4], color=gyro_colors[i],
                         linewidth=0.7, alpha=0.3, label=f"{gyro_labels[i]} (raw)")
        axes[i + 3].set_ylabel("rad/s")
        axes[i + 3].legend(loc="upper right")
        axes[i + 3].grid(True, linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()


def _plot_wrench_axes(axes, t, data, labels_f, labels_t, force_colors, torque_colors,
                      linewidth=1.2, alpha=1.0, suffix=""):
    for i in range(3):
        axes[i].plot(t, data[:, i + 1], color=force_colors[i],
                     linewidth=linewidth, alpha=alpha, label=f"{labels_f[i]}{suffix}")
        axes[i].set_ylabel("N")
        axes[i].grid(True, linewidth=0.4, alpha=0.5)
    for i in range(3):
        axes[i + 3].plot(t, data[:, i + 4], color=torque_colors[i],
                         linewidth=linewidth, alpha=alpha, label=f"{labels_t[i]}{suffix}")
        axes[i + 3].set_ylabel("Nm")
        axes[i + 3].grid(True, linewidth=0.4, alpha=0.5)


def plot_force_sensor(data_raw_smoothed, data_calibrated):
    """One figure: sensor frame raw (faint) + calibrated."""
    t = data_calibrated[:, 0]
    force_colors  = ("tab:red",    "tab:green", "tab:blue")
    torque_colors = ("tab:orange", "tab:cyan",  "tab:purple")
    labels_f = ("Fx", "Fy", "Fz")
    labels_t = ("Tx", "Ty", "Tz")

    fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    fig.suptitle("Wrench — Sensor Frame")

    _plot_wrench_axes(axes, t, data_raw_smoothed, labels_f, labels_t,
                      force_colors, torque_colors, linewidth=0.8, alpha=0.35, suffix=" raw")
    _plot_wrench_axes(axes, t, data_calibrated, labels_f, labels_t,
                      force_colors, torque_colors, linewidth=1.2, suffix=" cal")

    for ax in axes:
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()


def plot_gripper_mass_force(t_imu, gripper_force_xyz):
    """Plot gripper inertial force (m*a) in gripper frame XYZ."""
    colors = ("tab:red", "tab:green", "tab:blue")
    labels = ("Fx_gripper", "Fy_gripper", "Fz_gripper")

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    fig.suptitle("Gripper Mass Force (m·a) — Gripper Frame")

    for i in range(3):
        axes[i].plot(t_imu, gripper_force_xyz[:, i], color=colors[i],
                     linewidth=1.2, label=labels[i])
        axes[i].set_ylabel("N")
        axes[i].legend(loc="upper right")
        axes[i].grid(True, linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()


def _plot_compensation_fit(t_sync, wrench_sync, accel_sync, comp):
    """Plot measured vs predicted gripper wrench to assess fit quality."""
    predicted = comp.gripper_wrench(accel_sync)   # (K, 6)
    labels = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
    units  = ("N", "N", "N", "Nm", "Nm", "Nm")
    colors = ("tab:red", "tab:green", "tab:blue",
              "tab:orange", "tab:cyan", "tab:purple")

    fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    fig.suptitle("Gripper Compensation Fit — Measured vs Predicted (sensor frame)")

    for i in range(6):
        axes[i].plot(t_sync, wrench_sync[:, i], color=colors[i],
                     linewidth=1.2, label=f"{labels[i]} measured")
        axes[i].plot(t_sync, predicted[:, i], color=colors[i],
                     linewidth=1.0, linestyle="--", alpha=0.8,
                     label=f"{labels[i]} predicted")
        axes[i].set_ylabel(units[i])
        axes[i].legend(loc="upper right")
        axes[i].grid(True, linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()


def plot_force_gripper(data_gripper):
    """One figure: calibrated wrench in gripper frame."""
    t = data_gripper[:, 0]
    force_colors  = ("tab:red",    "tab:green", "tab:blue")
    torque_colors = ("tab:orange", "tab:cyan",  "tab:purple")
    labels_f = ("Fx", "Fy", "Fz")
    labels_t = ("Tx", "Ty", "Tz")

    fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    fig.suptitle("Wrench — Gripper Frame")

    _plot_wrench_axes(axes, t, data_gripper, labels_f, labels_t,
                      force_colors, torque_colors, linewidth=1.2)

    for ax in axes:
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()


# ==========================================
# --- 4. ENTRY POINT ---
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Fit gripper compensation from a gripper-only calibration bag."
    )
    parser.add_argument(
        "bag_path",
        nargs="?",
        default=str(BAG_PATH),
        help=f"Path to the gripper-only calibration rosbag (default: {BAG_PATH})",
    )
    parser.add_argument(
        "--calib-bag",
        default=str(CALIB_BAG_PATH),
        help=f"Path to the no-load sensor bias rosbag (default: {CALIB_BAG_PATH})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for gripper_comp.npz (default: next to the data bag).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib plots.",
    )
    args = parser.parse_args()

    bag = Path(args.bag_path).expanduser()
    out_path = Path(args.out).expanduser() if args.out else bag.parent / "gripper_comp.npz"

    # --- Step 1: sensor bias offset (no-load bag) ---
    print("Step 1: Computing sensor bias offset …")
    offset = WrenchCalibrator(bag_path=args.calib_bag).compute_offset()

    # --- Step 2: Load IMU and transform to sensor frame ---
    print("\nStep 2: Loading right-camera IMU …")
    imu_raw = load_imu_right(bag)          # (N, 7): [t, ax, ay, az, gx, gy, gz]

    acc  = imu_raw[:, 1:4]
    gyro = imu_raw[:, 4:7]

    # IMU frame → gripper frame
    obj_frame = ObjectFrame("imu_right")
    accel_gripper, gyro_gripper = obj_frame.process_data(acc, gyro)
    imu_gripper = np.hstack([imu_raw[:, :1], accel_gripper, gyro_gripper])

    # gripper frame → sensor frame  (R_SENSOR2GRIPPER.T maps gripper→sensor)
    R_GRIPPER2SENSOR = R_SENSOR2GRIPPER.T
    accel_sensor = (R_GRIPPER2SENSOR @ accel_gripper.T).T  # (N, 3)

    # --- Step 3: Load wrench and subtract sensor bias ---
    print("\nStep 3: Loading wrench …")
    force_raw = load_force(bag)            # (M, 7): [t, Fx, Fy, Fz, Tx, Ty, Tz]

    force_smoothed = remove_spikes(force_raw)
    force_net = force_smoothed.copy()
    force_net[:, 1:] -= offset             # bias-subtracted, sensor frame

    # Rotate wrench to gripper frame (for visualisation only)
    force_gripper = force_net.copy()
    force_gripper[:, 1:4] = (R_SENSOR2GRIPPER @ force_net[:, 1:4].T).T
    force_gripper[:, 4:7] = (R_SENSOR2GRIPPER @ force_net[:, 4:7].T).T

    # --- Step 4: Synchronise IMU → force timestamps via interpolation ---
    print("\nStep 4: Synchronising IMU to force timestamps …")
    t_force = force_net[:, 0]
    t_imu   = imu_raw[:, 0]

    # Clamp to overlapping time window
    t_start = max(t_force[0], t_imu[0])
    t_end   = min(t_force[-1], t_imu[-1])
    mask_f  = (t_force >= t_start) & (t_force <= t_end)
    t_sync  = t_force[mask_f]

    accel_sync = np.column_stack([
        np.interp(t_sync, t_imu, accel_sensor[:, i]) for i in range(3)
    ])                                     # (K, 3) acceleration in sensor frame
    wrench_sync = force_net[mask_f, 1:]   # (K, 6)

    print(f"  Synced {len(t_sync)} samples over {t_end - t_start:.2f} s")

    # --- Step 5: Fit gripper compensator ---
    print("\nStep 5: Fitting GripperCompensator …")
    comp = GripperCompensator(R_SENSOR2GRIPPER)
    comp.fit(accel_sync, wrench_sync)

    # --- Step 6: Save ---
    comp.save(out_path)
    print(f"\n{comp.summary()}")

    # --- Step 7: Diagnostic plots ---
    gripper_force_xyz = GRIPPER_MASS * accel_gripper   # reference using known mass

    if not args.no_plot:
        plot_imu_gripper(imu_raw, imu_gripper)
        plot_force_sensor(force_smoothed, force_net)
        plot_force_gripper(force_gripper)
        plot_gripper_mass_force(imu_raw[:, 0], gripper_force_xyz)
        _plot_compensation_fit(t_sync, wrench_sync, accel_sync, comp)
        plt.show()


if __name__ == "__main__":
    main()
