import glob
import os

import matplotlib
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
# Set this to True if you want to save the plot without a popup window appearing
HEADLESS_MODE = False
# Discard errors larger than this (in cm) to remove tracking "jumps" from stats
OUTLIER_THRESHOLD_CM = 100.0
TARGET_SAMPLES_PER_JOINT = 1000
OCCLUDED_SAMPLES_PER_CAMERA_PER_JOINT = 500
ACTIVE_JOINT_IDS = [0, 5, 6, 7, 8, 9, 10, 11, 12]

if HEADLESS_MODE:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


recordings_dir = "/home/pdz_cobot_lab/Videos/robot_recordings"
csv_files = glob.glob(os.path.join(recordings_dir, "hmr_validation_*.csv"))

if not csv_files:
    print("No CSV files found in the recordings directory.")
    exit()

latest_csv = max(csv_files, key=os.path.getctime)
print(f"Analyzing Data from: {latest_csv}")

df = pd.read_csv(latest_csv)
df['Error_cm'] = df['Error_m'] * 100
initial_count = len(df)

df = df[df['Error_cm'] < OUTLIER_THRESHOLD_CM].copy()
filtered_count = len(df)
if initial_count != filtered_count:
    print(f"Filtered out {initial_count - filtered_count} outliers (> {OUTLIER_THRESHOLD_CM}cm)")

mapping = {
    0: "nose", 5: "L shoulder", 6: "R shoulder", 7: "L elbow", 8: "R elbow",
    9: "L wrist", 10: "R wrist", 11: "L hip", 12: "R hip",
    13: "L knee", 14: "R knee", 15: "L ankle", 16: "R ankle"
}

if 'Error_Mode' in df.columns:
    baseline_df = df[df['Error_Mode'] == 'YOLO_vs_YOLO'].copy()
    occlusion_df = df[df['Error_Mode'] == 'HMR2_vs_YOLO'].copy()
else:
    print("Older CSV format detected. Vector uncertainty analysis needs a new validation CSV.")
    baseline_df = df[(df['Cam1_Source'] == 'YOLO') & (df['Cam2_Source'] == 'YOLO')].copy()
    occlusion_mask = (
        ((df['Cam1_Source'] == 'HMR2') & (df['Cam2_Source'] == 'YOLO')) |
        ((df['Cam1_Source'] == 'YOLO') & (df['Cam2_Source'] == 'HMR2'))
    )
    occlusion_df = df[occlusion_mask].copy()


def balance_samples_per_joint(data_df, name):
    if data_df.empty:
        return data_df

    counts = data_df.groupby('Joint_ID').size()
    available_joint_ids = [
        joint_id
        for joint_id in ACTIVE_JOINT_IDS
        if counts.get(joint_id, 0) > 0
    ]
    missing_joint_ids = [
        joint_id
        for joint_id in ACTIVE_JOINT_IDS
        if counts.get(joint_id, 0) == 0
    ]

    if missing_joint_ids:
        missing_names = [mapping.get(joint_id, f"ID {joint_id}") for joint_id in missing_joint_ids]
        print(f"WARNING: {name} has no samples for: {', '.join(missing_names)}")

    if not available_joint_ids:
        return data_df.iloc[0:0].copy()

    samples_per_joint = min(
        TARGET_SAMPLES_PER_JOINT,
        min(int(counts[joint_id]) for joint_id in available_joint_ids),
    )

    if samples_per_joint <= 0:
        return data_df.iloc[0:0].copy()

    balanced_df = (
        data_df[data_df['Joint_ID'].isin(available_joint_ids)]
        .groupby('Joint_ID', group_keys=False)
        .head(samples_per_joint)
        .copy()
    )

    print(
        f"{name}: using {samples_per_joint} samples per joint "
        f"across {len(available_joint_ids)} joints."
    )
    return balanced_df


def balance_occlusion_samples(data_df):
    if data_df.empty or 'Error_Bucket' not in data_df.columns:
        return balance_samples_per_joint(data_df, "OCCLUSION (HMR2 vs YOLO)")

    occlusion_buckets = ["CAM1_HMR2_CAM2_YOLO", "CAM1_YOLO_CAM2_HMR2"]
    data_df = data_df[data_df['Error_Bucket'].isin(occlusion_buckets)].copy()
    if data_df.empty:
        return data_df

    counts = data_df.groupby(['Joint_ID', 'Error_Bucket']).size()
    available_joint_ids = []
    missing = []

    for joint_id in ACTIVE_JOINT_IDS:
        bucket_counts = [counts.get((joint_id, bucket), 0) for bucket in occlusion_buckets]
        if all(count > 0 for count in bucket_counts):
            available_joint_ids.append(joint_id)
        else:
            missing_parts = [
                bucket
                for bucket, count in zip(occlusion_buckets, bucket_counts)
                if count == 0
            ]
            missing.append((joint_id, missing_parts))

    for joint_id, missing_parts in missing:
        joint_name = mapping.get(joint_id, f"ID {joint_id}")
        print(f"WARNING: OCCLUSION missing {joint_name}: {', '.join(missing_parts)}")

    if not available_joint_ids:
        return data_df.iloc[0:0].copy()

    samples_per_bucket = min(
        OCCLUDED_SAMPLES_PER_CAMERA_PER_JOINT,
        min(
            int(counts[(joint_id, bucket)])
            for joint_id in available_joint_ids
            for bucket in occlusion_buckets
        ),
    )

    balanced_df = (
        data_df[data_df['Joint_ID'].isin(available_joint_ids)]
        .groupby(['Joint_ID', 'Error_Bucket'], group_keys=False)
        .head(samples_per_bucket)
        .copy()
    )

    print(
        f"OCCLUSION (HMR2 vs YOLO): using {samples_per_bucket} samples per camera direction "
        f"per joint across {len(available_joint_ids)} joints."
    )
    return balanced_df


baseline_df = balance_samples_per_joint(baseline_df, "BASELINE (YOLO vs YOLO)")
occlusion_df = balance_occlusion_samples(occlusion_df)


def get_stats(data_df, name):
    if data_df.empty:
        return None, pd.DataFrame()

    overall_mean = data_df['Error_cm'].mean()
    print(f"=== {name} ===")
    print(f"Overall Mean: {overall_mean:.2f} cm | Samples: {len(data_df)}")

    stats = data_df.groupby('Joint_ID')['Error_cm'].agg(['mean', 'std', 'count']).reset_index()
    return overall_mean, stats


def uncertainty_summary(data_df, name):
    vector_cols = ['Error_X_m', 'Error_Y_m', 'Error_Z_m']
    if data_df.empty or not all(col in data_df.columns for col in vector_cols):
        return pd.DataFrame()

    rows = []
    print(f"\n=== {name} UNCERTAINTY SHAPE ===")

    for joint_id, group in data_df.groupby('Joint_ID'):
        if len(group) < 3:
            continue

        errors_cm = group[vector_cols].to_numpy(dtype=float) * 100.0
        bias = errors_cm.mean(axis=0)
        cov = np.cov(errors_cm, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        axes_1sigma = np.sqrt(np.maximum(eigvals, 0.0))
        anisotropy = axes_1sigma[0] / max(axes_1sigma[-1], 1e-6)
        rms_3d = np.sqrt(np.mean(np.sum(errors_cm**2, axis=1)))
        joint_name = mapping.get(int(joint_id), f"ID {int(joint_id)}")

        rows.append({
            "Joint_ID": joint_id,
            "Joint_Name": joint_name,
            "Samples": len(group),
            "Bias_X_cm": bias[0],
            "Bias_Y_cm": bias[1],
            "Bias_Z_cm": bias[2],
            "RMS_3D_cm": rms_3d,
            "Axis1_1sigma_cm": axes_1sigma[0],
            "Axis2_1sigma_cm": axes_1sigma[1],
            "Axis3_1sigma_cm": axes_1sigma[2],
            "Anisotropy": anisotropy,
            "Axis1_Direction_X": eigvecs[0, 0],
            "Axis1_Direction_Y": eigvecs[1, 0],
            "Axis1_Direction_Z": eigvecs[2, 0],
        })

        print(
            f"{joint_name:>12}: bias=[{bias[0]:6.2f}, {bias[1]:6.2f}, {bias[2]:6.2f}] cm | "
            f"1sigma axes=[{axes_1sigma[0]:5.2f}, {axes_1sigma[1]:5.2f}, {axes_1sigma[2]:5.2f}] cm | "
            f"anisotropy={anisotropy:4.2f} | n={len(group)}"
        )

    return pd.DataFrame(rows)


def draw_cov_ellipse(ax, data_2d, color, label):
    if len(data_2d) < 3:
        return

    mean = data_2d.mean(axis=0)
    cov = np.cov(data_2d, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    axes = 2.0 * np.sqrt(np.maximum(eigvals, 0.0))
    ellipse = Ellipse(
        xy=mean,
        width=axes[0],
        height=axes[1],
        angle=angle,
        edgecolor=color,
        facecolor='none',
        linewidth=2.0,
        label=label,
    )
    ax.add_patch(ellipse)
    ax.scatter(mean[0], mean[1], color=color, marker='x', s=80)


def draw_joint_uncertainty_ellipses(data_df, base_path):
    vector_cols = ['Error_X_m', 'Error_Y_m', 'Error_Z_m']
    if data_df.empty or not all(col in data_df.columns for col in vector_cols):
        return

    projection_specs = [
        (0, 1, "X error (cm)", "Y error (cm)", "XY"),
        (0, 2, "X error (cm)", "Z error (cm)", "XZ"),
        (1, 2, "Y error (cm)", "Z error (cm)", "YZ"),
    ]

    joint_groups = [
        (joint_id, group)
        for joint_id, group in data_df.groupby('Joint_ID')
        if len(group) >= 3
    ]
    if not joint_groups:
        return

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(joint_groups)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (i, j, xlabel, ylabel, title) in zip(axes, projection_specs):
        for color, (joint_id, group) in zip(colors, joint_groups):
            errors_cm = group[vector_cols].to_numpy(dtype=float) * 100.0
            projected = errors_cm[:, [i, j]]
            joint_name = mapping.get(int(joint_id), f"ID {int(joint_id)}")
            draw_cov_ellipse(ax, projected, color, joint_name)

        ax.axhline(0.0, color='gray', linewidth=0.8)
        ax.axvline(0.0, color='gray', linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Per-Joint Occlusion Uncertainty: {title}")
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.axis('equal')

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    ellipse_path = f"{base_path}_per_joint_uncertainty_ellipses.png"
    fig.savefig(ellipse_path, dpi=300)
    print(f"SUCCESS: Per-joint uncertainty ellipse plot saved to {ellipse_path}")


def draw_overall_uncertainty_ellipses(data_df, base_path):
    vector_cols = ['Error_X_m', 'Error_Y_m', 'Error_Z_m']
    if data_df.empty or not all(col in data_df.columns for col in vector_cols) or len(data_df) < 3:
        return

    errors_cm = data_df[vector_cols].to_numpy(dtype=float) * 100.0
    projection_specs = [
        (0, 1, "X error (cm)", "Y error (cm)", "XY"),
        (0, 2, "X error (cm)", "Z error (cm)", "XZ"),
        (1, 2, "Y error (cm)", "Z error (cm)", "YZ"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (i, j, xlabel, ylabel, title) in zip(axes, projection_specs):
        projected = errors_cm[:, [i, j]]
        ax.scatter(projected[:, 0], projected[:, 1], s=10, alpha=0.18, color='tab:red')
        draw_cov_ellipse(ax, projected, 'black', 'pooled 1-sigma covariance ellipse')
        ax.axhline(0.0, color='gray', linewidth=0.8)
        ax.axvline(0.0, color='gray', linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Pooled Occlusion Error Projection: {title}")
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.axis('equal')

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    ellipse_path = f"{base_path}_pooled_uncertainty_ellipses.png"
    fig.savefig(ellipse_path, dpi=300)
    print(f"SUCCESS: Pooled uncertainty ellipse plot saved to {ellipse_path}")


base_mean, base_joint_stats = get_stats(baseline_df, "BASELINE (YOLO vs YOLO)")
occ_mean, occ_joint_stats = get_stats(occlusion_df, "OCCLUSION (HMR2 vs YOLO)")

base_path = os.path.splitext(latest_csv)[0]

occ_uncertainty = uncertainty_summary(occlusion_df, "OCCLUSION")
if not occ_uncertainty.empty:
    summary_path = f"{base_path}_uncertainty_summary.csv"
    occ_uncertainty.to_csv(summary_path, index=False)
    print(f"\nSUCCESS: Uncertainty summary saved to {summary_path}")

if not occ_joint_stats.empty:
    plt.figure(figsize=(12, 7))

    if base_joint_stats.empty:
        plot_df = occ_joint_stats.rename(columns={'mean': 'mean_occ', 'std': 'std_occ', 'count': 'count_occ'})
        plot_df['mean_base'] = np.nan
    else:
        plot_df = pd.merge(occ_joint_stats, base_joint_stats, on='Joint_ID', how='left', suffixes=('_occ', '_base'))
    x_labels = [mapping.get(int(jid), f"ID {int(jid)}") for jid in plot_df['Joint_ID']]

    plt.bar(
        x_labels,
        plot_df['mean_occ'],
        yerr=plot_df['std_occ'],
        capsize=5,
        color='salmon',
        alpha=0.8,
        edgecolor='black',
        label=f'HMR2 Guessing Error ({occ_mean:.2f} cm)',
    )

    if base_mean is not None:
        plt.plot(
            x_labels,
            plot_df['mean_base'],
            color='green',
            marker='o',
            linestyle='--',
            linewidth=2,
            markersize=8,
            label=f'System Baseline ({base_mean:.2f} cm)',
        )

    plt.title("Joint Tracking Accuracy: Baseline vs. Occlusion")
    plt.ylabel("Error (cm)")
    plt.xticks(rotation=35, ha='right')
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()

    plot_path = f"{base_path}_analysis_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"SUCCESS: Chart saved to {plot_path}")

    draw_joint_uncertainty_ellipses(occlusion_df, base_path)
    draw_overall_uncertainty_ellipses(occlusion_df, base_path)

    if not HEADLESS_MODE:
        plt.show()
else:
    print("No data available to plot.")
