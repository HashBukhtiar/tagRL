"""
figures/plot_results.py — Generate all report figures from CSV and TensorBoard data.

Usage:
    python figures/plot_results.py [--results_dir DIR] [--tb_log_dir DIR]
                                   [--output_dir DIR]

Figures produced:
  1. co_evolution.png / .pdf  — main result: two diverging total-tags curves
  2. training_curves.png / .pdf  — ep_rew_mean for both agents over training
  3. episode_length.png / .pdf   — ep_len_mean over training (sanity check; always ~500)

All figures are saved as both .png (for presentations/slides) and .pdf (for the
LaTeX report), following the recommendation in CLAUDE.md to separate data
collection from figure styling.

Design note: all figures use a minimal, publication-quality style.  Serif font,
thin grid lines, no top/right spines.  Colours are chosen to be distinguishable
in greyscale (solid vs dashed) and for accessibility (blue vs orange).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "legend.fontsize":    10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# Colour palette — distinguishable in greyscale via linestyle as well
RUNNER_COLOR = "#2166ac"   # blue
TAGGER_COLOR = "#d6604d"   # orange-red


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate report figures from evaluation CSVs and TensorBoard logs."
    )
    p.add_argument("--results_dir", type=str, default="results",
                   help="Directory containing evaluation CSVs (default: results/).")
    p.add_argument("--tb_log_dir",  type=str, default="./tensorboard_logs/",
                   help="Root directory for TensorBoard logs (default: ./tensorboard_logs/).")
    p.add_argument("--output_dir",  type=str, default="figures/output",
                   help="Directory to save figure files (default: figures/output/).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Saving helper
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, output_dir: str, name: str) -> None:
    """Save figure as both .png and .pdf."""
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 1: Co-evolution plot
# ---------------------------------------------------------------------------

def plot_coevolution(results_dir: str, output_dir: str) -> None:
    """
    Main result figure.

    Two curves on one axis:
      - Blue (solid):  latest runner vs historical taggers
                       → tags conceded should DECREASE (runner improved, evades more)
      - Red (dashed):  latest tagger vs historical runners
                       → tags scored should INCREASE (tagger improved, catches more)

    If both trends hold simultaneously, both agents genuinely co-evolved.
    Episodes always run for MAX_STEPS=500 steps, so total_tags per episode
    is the meaningful metric (episode duration is a constant, not useful here).

    The shaded bands show ±1 std across evaluation episodes, giving a sense
    of episode-to-episode variability (wide bands = high variance in outcomes).
    """
    path_A = os.path.join(results_dir, "runner_vs_historical_taggers.csv")
    path_B = os.path.join(results_dir, "tagger_vs_historical_runners.csv")

    if not os.path.exists(path_A) or not os.path.exists(path_B):
        print(f"  [skip] Co-evolution CSVs not found in '{results_dir}'. "
              f"Run eval/evaluate.py first.")
        return

    df_A = pd.read_csv(path_A)   # latest runner vs historical taggers
    df_B = pd.read_csv(path_B)   # latest tagger vs historical runners

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Curve A: runner vs historical taggers (tags conceded should decrease)
    ax.plot(
        df_A["snapshot_cycle"], df_A["mean_tags"],
        color=RUNNER_COLOR, linestyle="-", linewidth=2,
        marker="o", markersize=4,
        label="Latest runner vs historical tagger (tags conceded — lower is better)",
    )
    ax.fill_between(
        df_A["snapshot_cycle"],
        df_A["mean_tags"] - df_A["std_tags"],
        df_A["mean_tags"] + df_A["std_tags"],
        color=RUNNER_COLOR, alpha=0.15,
    )

    # Curve B: tagger vs historical runners (tags scored should increase)
    ax.plot(
        df_B["snapshot_cycle"], df_B["mean_tags"],
        color=TAGGER_COLOR, linestyle="--", linewidth=2,
        marker="s", markersize=4,
        label="Latest tagger vs historical runner (tags scored — higher is better)",
    )
    ax.fill_between(
        df_B["snapshot_cycle"],
        df_B["mean_tags"] - df_B["std_tags"],
        df_B["mean_tags"] + df_B["std_tags"],
        color=TAGGER_COLOR, alpha=0.15,
    )

    ax.set_xlabel("Historical snapshot (training cycle)")
    ax.set_ylabel("Mean tags per episode")
    ax.set_title("Co-evolution: diverging tag counts confirm genuine improvement")
    ax.legend(loc="center right")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.4, alpha=0.4)

    fig.tight_layout()
    save_fig(fig, output_dir, "co_evolution")
    plt.close(fig)


# ---------------------------------------------------------------------------
# TensorBoard reader
# ---------------------------------------------------------------------------

def read_tb_scalar(log_dir: str, tag: str) -> tuple:
    """
    Read a scalar series from a TensorBoard event file.
    Returns (steps_array, values_array) or (None, None) if unavailable.

    Requires the 'tensorboard' package (installed as part of SB3 deps).
    Falls back gracefully if the log directory is missing or the tag is absent.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator, TENSORS, SCALARS,
        )
    except ImportError:
        print("  [warn] tensorboard package not available — skipping TB plots.")
        return None, None

    if not os.path.isdir(log_dir):
        print(f"  [warn] TensorBoard log directory not found: {log_dir}")
        return None, None

    ea = EventAccumulator(log_dir, size_guidance={SCALARS: 0})
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        return None, None

    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def find_tb_run_dirs(tb_log_dir: str, role: str) -> list:
    """
    SB3 creates subdirectories like 'runner_1/', 'runner_2/', etc.
    Return all matching directories sorted by modification time.
    """
    import glob as _glob
    pattern = os.path.join(tb_log_dir, f"{role}_*")
    dirs    = sorted(_glob.glob(pattern))
    return dirs


def load_tb_scalar_merged(tb_log_dir: str, role: str, tag: str):
    """
    Load a scalar tag from all TensorBoard run directories for a given role,
    merging them into a single (steps, values) array ordered by step.
    SB3 creates a new subdirectory each time .learn() is first called with a
    new tb_log_name, so there may be multiple run directories to merge.
    With reset_num_timesteps=False this should be a single directory, but
    we handle the multi-run case defensively.
    """
    run_dirs = find_tb_run_dirs(tb_log_dir, role)
    if not run_dirs:
        return None, None

    all_steps, all_values = [], []
    for d in run_dirs:
        s, v = read_tb_scalar(d, tag)
        if s is not None:
            all_steps.append(s)
            all_values.append(v)

    if not all_steps:
        return None, None

    steps  = np.concatenate(all_steps)
    values = np.concatenate(all_values)
    order  = np.argsort(steps)
    return steps[order], values[order]


# ---------------------------------------------------------------------------
# Figure 2: Training curves (ep_rew_mean)
# ---------------------------------------------------------------------------

def plot_training_curves(tb_log_dir: str, output_dir: str) -> None:
    """
    Episode reward mean for both agents over training.
    Tagger reward should trend upward (catching runner more reliably).
    Runner reward should trend upward (surviving longer).
    Both going up simultaneously is consistent with co-evolution (the total
    reward pool is not fixed — more episodes can go to max_steps).
    """
    tag = "rollout/ep_rew_mean"

    runner_steps, runner_values = load_tb_scalar_merged(tb_log_dir, "runner", tag)
    tagger_steps, tagger_values = load_tb_scalar_merged(tb_log_dir, "tagger", tag)

    if runner_steps is None and tagger_steps is None:
        print(f"  [skip] No TensorBoard data found for '{tag}'. "
              f"Run agents/train.py first.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    if runner_steps is not None:
        ax.plot(runner_steps, runner_values,
                color=RUNNER_COLOR, linewidth=1.5, label="Runner  ep_rew_mean")
    if tagger_steps is not None:
        ax.plot(tagger_steps, tagger_values,
                color=TAGGER_COLOR, linewidth=1.5, linestyle="--",
                label="Tagger  ep_rew_mean")

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Training curves — episode reward mean")
    ax.legend()
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    fig.tight_layout()
    save_fig(fig, output_dir, "training_curves")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Episode length over training
# ---------------------------------------------------------------------------

def plot_episode_length(tb_log_dir: str, output_dir: str) -> None:
    """
    ep_len_mean for both agents over training.

    Sanity-check figure.  With true-tag role-switching, episodes always run to
    MAX_STEPS=500 steps (they never terminate on a catch), so both curves should
    converge to ~500 throughout training.  A flat line at 500 confirms that the
    termination logic is correct and no early exits are occurring.
    """
    tag = "rollout/ep_len_mean"

    runner_steps, runner_values = load_tb_scalar_merged(tb_log_dir, "runner", tag)
    tagger_steps, tagger_values = load_tb_scalar_merged(tb_log_dir, "tagger", tag)

    if runner_steps is None and tagger_steps is None:
        print(f"  [skip] No TensorBoard data found for '{tag}'.")
        return

    from env.tag_env import MAX_STEPS

    fig, ax = plt.subplots(figsize=(7, 4))

    if runner_steps is not None:
        ax.plot(runner_steps, runner_values,
                color=RUNNER_COLOR, linewidth=1.5, label="Runner  ep_len_mean")
    if tagger_steps is not None:
        ax.plot(tagger_steps, tagger_values,
                color=TAGGER_COLOR, linewidth=1.5, linestyle="--",
                label="Tagger  ep_len_mean")

    ax.axhline(MAX_STEPS, color="grey", linestyle=":", linewidth=1,
               label=f"Max steps ({MAX_STEPS})")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mean episode length (steps)")
    ax.set_title("Episode length over training (sanity check — should be flat at 500)")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    fig.tight_layout()
    save_fig(fig, output_dir, "episode_length")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"\nGenerating figures → '{args.output_dir}/'")
    print(f"  results_dir : {args.results_dir}")
    print(f"  tb_log_dir  : {args.tb_log_dir}")

    print("\n[Figure 1] Co-evolution plot")
    plot_coevolution(args.results_dir, args.output_dir)

    print("\n[Figure 2] Training curves (ep_rew_mean)")
    plot_training_curves(args.tb_log_dir, args.output_dir)

    print("\n[Figure 3] Episode length (ep_len_mean)")
    plot_episode_length(args.tb_log_dir, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
