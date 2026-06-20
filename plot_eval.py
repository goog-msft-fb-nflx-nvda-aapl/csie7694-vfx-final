"""
Generate evaluation charts for VideoCoF temporal consistency analysis.
Run on gsm-gpu: python plot_eval.py --results /home/jtan/vfx/eval_results.jsonl
"""
import argparse
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_length_trend(rows, out="eval_length_trend.png"):
    # Filter long-video series (same video, different lengths)
    tags = ["local_style_daiyu_33", "local_style_128", "local_style_256", "local_style_512"]
    labels = ["33f", "128f", "256f", "512f"]
    by_tag = {r["tag"]: r for r in rows}

    frames = []
    flicker, flow, drift = [], [], []
    used_labels = []
    for tag, label in zip(tags, labels):
        if tag in by_tag:
            r = by_tag[tag]
            frames.append(r["num_frames"])
            flicker.append(r["flicker"])
            flow.append(r["flow_smoothness"])
            drift.append(r["clip_drift"] if r["clip_drift"] is not None else 0)
            used_labels.append(label)

    x = np.arange(len(used_labels))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("VideoCoF Temporal Consistency vs. Edit Length\n(local_style on daiyu_529frames.mp4)",
                 fontsize=12, fontweight="bold")

    for ax, vals, ylabel, color in zip(
        axes,
        [flicker, flow, drift],
        ["Flicker (L2 diff) ↓", "Flow Smoothness ↓", "CLIP Drift ↓"],
        ["#e74c3c", "#3498db", "#2ecc71"]
    ):
        ax.plot(x, vals, "o-", color=color, linewidth=2.5, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(used_labels)
        ax.set_xlabel("Edit length (frames)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.1f}", (x[i], v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


def plot_task_comparison(rows, out="eval_task_comparison.png"):
    tags = ["obj_rem_33", "obj_add_33", "obj_swap_33", "local_style_33"]
    labels = ["Obj Remove", "Obj Add", "Obj Swap", "Local Style"]
    by_tag = {r["tag"]: r for r in rows}

    metrics = {"Flicker↓": [], "Flow↓": [], "CLIP Drift↓": [], "Edit Coverage": []}
    used_labels = []
    for tag, label in zip(tags, labels):
        if tag in by_tag:
            r = by_tag[tag]
            metrics["Flicker↓"].append(r["flicker"])
            metrics["Flow↓"].append(r["flow_smoothness"])
            metrics["CLIP Drift↓"].append(r["clip_drift"] if r["clip_drift"] is not None else 0)
            metrics["Edit Coverage"].append(r["edit_coverage"])
            used_labels.append(label)

    x = np.arange(len(used_labels))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle("VideoCoF Per-Task Temporal Consistency Metrics (33 frames)",
                 fontsize=12, fontweight="bold")

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for ax, (metric, vals), color in zip(axes, metrics.items(), colors):
        bars = ax.bar(x, vals, color=color, alpha=0.8, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(used_labels, rotation=12, fontsize=9)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(vals),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="eval_results.jsonl")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    rows = load(args.results)
    print(f"Loaded {len(rows)} eval entries")

    import os
    plot_length_trend(rows, out=os.path.join(args.outdir, "eval_length_trend.png"))
    plot_task_comparison(rows, out=os.path.join(args.outdir, "eval_task_comparison.png"))
    print("Done.")


if __name__ == "__main__":
    main()
