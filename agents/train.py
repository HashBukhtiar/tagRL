"""
agents/train.py — Alternating PPO self-play training loop.

Usage:
    python agents/train.py [--num_cycles N] [--steps_per_cycle S]
                           [--snapshot_freq F] [--seed SEED]
                           [--resume_from CYCLE]

Key design decisions documented here for the report:

1. Alternating (not simultaneous) training
   In simultaneous updates both policies change at every step, making the
   environment non-stationary from each agent's perspective.  Non-stationarity
   violates the Markov property that PPO assumes and can cause divergent or
   cycling behaviour.  Alternating updates freeze one agent while the other
   trains, giving the training agent a stationary opponent — a proper MDP.
   The tradeoff is that each agent trains against a slightly lagged opponent,
   but empirically this is far more stable than simultaneous updates.

2. Opponent is injected via set_opponent(), not re-created
   TaggerEnv and RunnerEnv hold a reference to an opponent model and call
   model.predict() inside step().  When we update the frozen model between
   cycles we call env.set_opponent(new_model), which modifies the env object
   in place.  Because SB3 wraps our env in a DummyVecEnv that holds a reference
   to the same object, the change propagates automatically — no need to
   reconstruct the PPO model or its environment.

3. Snapshot at cycle 0 (random baseline)
   We save an initial snapshot before any training.  This gives the evaluator
   a genuine "random opponent" baseline, making the co-evolution plot's
   x-axis start from a meaningful reference point.

4. reset_num_timesteps=False
   Keeps SB3's internal timestep counter running continuously across learn()
   calls.  This means TensorBoard logs show a single continuous curve rather
   than resetting to 0 every cycle — much easier to read.

5. steps_per_cycle vs n_steps
   n_steps (PPO rollout buffer size) is the number of environment steps
   collected before each gradient update.  steps_per_cycle must be a multiple
   of n_steps so that learn() completes an integer number of update rounds.
   Default: n_steps=1024, steps_per_cycle=2048 → 2 PPO updates per cycle.
"""

import argparse
import os
import sys
import time

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from env.tag_env import TaggerEnv, RunnerEnv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train tagger and runner via alternating PPO self-play."
    )
    p.add_argument(
        "--num_cycles", type=int, default=200,
        help="Number of alternating training cycles (default: 200). "
             "Total env steps ≈ 2 × num_cycles × steps_per_cycle.",
    )
    p.add_argument(
        "--steps_per_cycle", type=int, default=2048,
        help="Environment steps per agent per cycle (default: 2048). "
             "Must be a multiple of --n_steps.",
    )
    p.add_argument(
        "--n_steps", type=int, default=1024,
        help="PPO rollout buffer size — steps collected before each gradient "
             "update (default: 1024).  steps_per_cycle must be a multiple.",
    )
    p.add_argument(
        "--snapshot_freq", type=int, default=8,
        help="Save snapshots every N cycles (default: 8). "
             "With 200 cycles this gives 25 snapshots — within the 20-30 target.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    p.add_argument(
        "--snapshots_dir", type=str, default="snapshots",
        help="Directory to save policy snapshots (default: snapshots/).",
    )
    p.add_argument(
        "--tb_log_dir", type=str, default="./tensorboard_logs/",
        help="Root directory for TensorBoard logs (default: ./tensorboard_logs/).",
    )
    p.add_argument(
        "--resume_from", type=int, default=0,
        help="Resume training from this cycle number. Loads tagger_NNNN and "
             "runner_NNNN snapshots from --snapshots_dir and starts from cycle+1 "
             "(default: 0, i.e. fresh start).",
    )
    p.add_argument(
        "--check_env", action="store_true",
        help="Run SB3's check_env() on both envs before training and exit.",
    )
    p.add_argument(
        "--render", action="store_true",
        help="Open a pygame window and play one episode after each snapshot checkpoint. "
             "Slows training slightly at checkpoint intervals only — no cost between them.",
    )
    p.add_argument(
        "--replay", action="store_true",
        help="After training completes, play back one episode per snapshot in cycle order "
             "so you can watch the agents evolve from random to trained.",
    )
    p.add_argument(
        "--render_fps", type=int, default=10,
        help="Frames per second for the live render/replay window (default: 10).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------

def make_ppo(env, n_steps: int, tb_log_dir: str, seed: int, role: str) -> PPO:
    """
    Create a PPO model with the hyperparameters from CLAUDE.md.

    ent_coef=0.01: entropy bonus keeps exploration alive during early training.
    Without it the policy collapses to near-deterministic behaviour before the
    opponent is interesting, and both agents get stuck in a local equilibrium.

    gamma=0.99: long discount horizon is appropriate because episodes run for
    MAX_STEPS=500 steps.  A shorter horizon (e.g. 0.95) would undervalue future
    rewards and make both agents overly myopic.
    """
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=tb_log_dir,
        seed=seed,
    )
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.snapshots_dir, exist_ok=True)
    os.makedirs(args.tb_log_dir, exist_ok=True)

    # --- Validate steps_per_cycle is a multiple of n_steps ---
    if args.steps_per_cycle % args.n_steps != 0:
        raise ValueError(
            f"steps_per_cycle ({args.steps_per_cycle}) must be a multiple of "
            f"n_steps ({args.n_steps}). "
            f"Try --steps_per_cycle {(args.steps_per_cycle // args.n_steps) * args.n_steps}."
        )

    # --- Set up optional live renderer ---
    # Imported here so pygame is never touched during a headless training run.
    renderer = None
    if args.render:
        from render.visualize import TagRenderer
        renderer = TagRenderer(fps=args.render_fps)
        print("Live render enabled — pygame window will appear at each checkpoint.")

    # --- Create environments ---
    tagger_env = TaggerEnv(seed=args.seed)
    runner_env = RunnerEnv(seed=args.seed + 1)  # different seed so they start at different positions

    if args.check_env:
        print("Running SB3 env checker on TaggerEnv...")
        check_env(tagger_env, warn=True)
        print("Running SB3 env checker on RunnerEnv...")
        check_env(runner_env, warn=True)
        print("[PASS] Both envs pass SB3 check_env()")
        return

    # --- Create or resume PPO models ---
    # Each model is tied to its own env.  SB3 wraps the env in a DummyVecEnv
    # internally; our set_opponent() calls modify the env object in-place, which
    # propagates to the wrapped env since they share the same object reference.
    start_cycle = args.resume_from + 1

    if args.resume_from > 0:
        tagger_snap = os.path.join(args.snapshots_dir, f"tagger_{args.resume_from:04d}")
        runner_snap = os.path.join(args.snapshots_dir, f"runner_{args.resume_from:04d}")
        for path in (tagger_snap + ".zip", runner_snap + ".zip"):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Resume snapshot not found: {path}\n"
                    f"Available snapshots in {args.snapshots_dir}/: "
                    + ", ".join(sorted(os.listdir(args.snapshots_dir)))
                )
        tagger_model = PPO.load(tagger_snap, env=tagger_env)
        runner_model = PPO.load(runner_snap, env=runner_env)
        print(f"Resumed from cycle {args.resume_from}: "
              f"{tagger_snap}.zip, {runner_snap}.zip")
    else:
        tagger_model = make_ppo(tagger_env, args.n_steps, args.tb_log_dir, args.seed,     "tagger")
        runner_model = make_ppo(runner_env, args.n_steps, args.tb_log_dir, args.seed + 1, "runner")

    print(f"\n{'='*60}")
    print(f"Training: cycles {start_cycle}–{args.num_cycles}, "
          f"{args.steps_per_cycle} steps/cycle/agent, "
          f"snapshots every {args.snapshot_freq} cycles")
    print(f"Remaining env steps (approx): "
          f"{2 * (args.num_cycles - args.resume_from) * args.steps_per_cycle:,}")
    print(f"{'='*60}\n")

    # --- Snapshot cycle 0 (untrained / random baseline) ---
    # Skipped when resuming — the existing snapshots are already on disk.
    if args.resume_from == 0:
        snap0_runner = os.path.join(args.snapshots_dir, "runner_0000")
        snap0_tagger = os.path.join(args.snapshots_dir, "tagger_0000")
        runner_model.save(snap0_runner)
        tagger_model.save(snap0_tagger)
        print(f"[Cycle 0] Saved initial (random) snapshots: "
              f"{snap0_runner}.zip, {snap0_tagger}.zip")

    # --- Alternating training ---
    train_start = time.time()

    for cycle in range(start_cycle, args.num_cycles + 1):
        cycle_start = time.time()

        # --- Step 1: Update runner, tagger frozen ---
        # The runner trains against the current tagger policy.  The tagger's
        # weights do not change during this step.
        runner_env.set_opponent(tagger_model)
        runner_model.learn(
            total_timesteps=args.steps_per_cycle,
            reset_num_timesteps=False,   # keep cumulative counter for TensorBoard
            tb_log_name="runner",
            progress_bar=False,
        )

        # --- Step 2: Update tagger, runner frozen ---
        # Now the tagger trains against the freshly updated runner policy.
        # The runner's weights do not change during this step.
        tagger_env.set_opponent(runner_model)
        tagger_model.learn(
            total_timesteps=args.steps_per_cycle,
            reset_num_timesteps=False,
            tb_log_name="tagger",
            progress_bar=False,
        )

        cycle_time = time.time() - cycle_start
        elapsed    = time.time() - train_start

        # --- Snapshot ---
        if cycle % args.snapshot_freq == 0:
            snap_runner = os.path.join(args.snapshots_dir, f"runner_{cycle:04d}")
            snap_tagger = os.path.join(args.snapshots_dir, f"tagger_{cycle:04d}")
            runner_model.save(snap_runner)
            tagger_model.save(snap_tagger)
            print(f"[Cycle {cycle:4d}/{args.num_cycles}] "
                  f"Snapshots saved  |  "
                  f"cycle: {cycle_time:.1f}s  elapsed: {elapsed/60:.1f}min")

            # --- Optional live render ---
            if renderer is not None:
                from render.visualize import render_episode
                render_episode(
                    tagger_model  = tagger_model,
                    runner_model  = runner_model,
                    renderer      = renderer,
                    seed          = args.seed + cycle,
                    deterministic = True,
                    label         = f"Cycle {cycle}/{args.num_cycles}",
                    record        = False,
                )

    # --- Always save the very last cycle if it wasn't already a snapshot cycle ---
    if args.num_cycles % args.snapshot_freq != 0:
        final_runner = os.path.join(args.snapshots_dir, f"runner_{args.num_cycles:04d}")
        final_tagger = os.path.join(args.snapshots_dir, f"tagger_{args.num_cycles:04d}")
        runner_model.save(final_runner)
        tagger_model.save(final_tagger)
        print(f"[Cycle {args.num_cycles}] Final snapshots saved.")

    if renderer is not None:
        renderer.quit()

    # --- Post-training replay ---
    # Loads every saved snapshot in cycle order and plays one episode each.
    # Training is already done so this has zero effect on training speed.
    if args.replay:
        import glob as _glob
        from render.visualize import TagRenderer, render_episode
        from stable_baselines3 import PPO as _PPO
        from env.tag_env import TaggerEnv as _TEnv, RunnerEnv as _REnv

        # Discover all snapshot pairs that exist on disk
        tagger_files = sorted(_glob.glob(os.path.join(args.snapshots_dir, "tagger_*.zip")))
        runner_files = sorted(_glob.glob(os.path.join(args.snapshots_dir, "runner_*.zip")))

        # Build a dict of cycle → path for each role, then keep only cycles
        # that have BOTH a tagger and a runner snapshot.
        def _parse(files, prefix):
            out = {}
            for f in files:
                base = os.path.basename(f)
                try:
                    cycle = int(base.replace(f"{prefix}_", "").replace(".zip", ""))
                    out[cycle] = f.replace(".zip", "")
                except ValueError:
                    pass
            return out

        t_snaps = _parse(tagger_files, "tagger")
        r_snaps = _parse(runner_files, "runner")
        common  = sorted(set(t_snaps) & set(r_snaps))

        if not common:
            print("[replay] No matching snapshot pairs found — skipping.")
        else:
            print(f"\n[replay] Playing back {len(common)} snapshots "
                  f"(cycles {common[0]}–{common[-1]}).  "
                  f"Close the window or press ESC to skip ahead.")

            dummy_t = _TEnv(seed=0)
            dummy_r = _REnv(seed=0)
            replay_renderer = TagRenderer(fps=args.render_fps)

            for cycle in common:
                print(f"  [replay] Cycle {cycle:4d} / {common[-1]}", end="\r", flush=True)
                t_model = _PPO.load(t_snaps[cycle], env=dummy_t)
                r_model = _PPO.load(r_snaps[cycle], env=dummy_r)

                render_episode(
                    tagger_model  = t_model,
                    runner_model  = r_model,
                    renderer      = replay_renderer,
                    seed          = args.seed + cycle,
                    deterministic = True,
                    label         = f"Cycle {cycle} / {common[-1]}",
                    record        = False,
                )

            replay_renderer.quit()
            print("\n[replay] Done.")

    total_time = time.time() - train_start
    print(f"\nTraining complete in {total_time/60:.1f} min.")
    print(f"Snapshots in:   {args.snapshots_dir}/")
    print(f"TensorBoard:    tensorboard --logdir {args.tb_log_dir}")


if __name__ == "__main__":
    main()
