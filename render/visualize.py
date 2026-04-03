"""
render/visualize.py — Pygame rendering for demo/presentation videos.

IMPORTANT: This file is NEVER imported during training.  It has no effect on
training speed or reproducibility.  Import it only for demos.

Usage:
    # Interactive window (no recording)
    python render/visualize.py --tagger snapshots/tagger_0000 --runner snapshots/runner_0000

    # Record a GIF
    python render/visualize.py --tagger snapshots/tagger_0200 --runner snapshots/runner_0200
                               --save output/late_stage.gif --fps 8

    # Render three stages back-to-back and save as separate GIFs
    python render/visualize.py --stages --snapshots_dir snapshots --save_dir output/

    # Replay all snapshots in order at a custom FPS (run any time after training)
    python render/visualize.py --replay --fps 12
    python render/visualize.py --replay --fps 4   # slower

Recommended stages for the presentation:
  1. Early  (cycle 0000): random movement — chaotic, agents ignore each other
  2. Mid    (cycle 0100): basic pursuit/evasion beginning to emerge
  3. Late   (cycle 0200): visible strategies — tagger corners runner, runner hugs walls
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.tag_env import GridState, ACTIONS, GRID_SIZE, MAX_STEPS, OBS_DIM

# ---------------------------------------------------------------------------
# Pygame is only imported here, never in training code
# ---------------------------------------------------------------------------
try:
    import pygame
except ImportError:
    print("[error] pygame is not installed.  Run: pip install pygame")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

CELL_SIZE    = 50           # pixels per grid cell
MARGIN       = 10           # pixel border around the grid
INFO_HEIGHT  = 40           # pixels for the info bar at the bottom
WINDOW_W     = GRID_SIZE * CELL_SIZE + 2 * MARGIN
WINDOW_H     = GRID_SIZE * CELL_SIZE + 2 * MARGIN + INFO_HEIGHT

# Colours (R, G, B)
BG_COLOR     = (245, 245, 245)
WALL_COLOR   = (60,  60,  60)
CELL_COLOR   = (210, 210, 210)
GRID_COLOR   = (185, 185, 185)
TAGGER_COLOR       = (210, 50,  50)    # red — active tagger
TAGGER_GRACE_COLOR = (180, 100, 200)   # purple — frozen tagger during grace period
RUNNER_COLOR       = (50,  100, 210)   # blue
TEXT_COLOR         = (40,  40,  40)
INFO_BG      = (230, 230, 230)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Render tag game episodes using Pygame."
    )
    # Single-episode mode
    p.add_argument("--tagger", type=str, default=None,
                   help="Path to tagger snapshot (without .zip).")
    p.add_argument("--runner", type=str, default=None,
                   help="Path to runner snapshot (without .zip).")
    # Three-stage mode
    p.add_argument("--stages", action="store_true",
                   help="Render three stages (early/mid/late) automatically.")
    # Replay-all mode
    p.add_argument("--replay", action="store_true",
                   help="Play one episode per snapshot in cycle order — "
                        "watch the agents evolve from cycle 0 to the last saved checkpoint.")
    p.add_argument("--snapshots_dir", type=str, default="snapshots",
                   help="Directory with snapshots for --stages / --replay mode.")
    p.add_argument("--save_dir", type=str, default=None,
                   help="Directory to save GIFs for --stages mode.")
    # Common options
    p.add_argument("--save", type=str, default=None,
                   help="Save path for a single GIF/MP4 (e.g. output/demo.gif).")
    p.add_argument("--fps", type=int, default=8,
                   help="Frames per second for display and recording (default: 8).")
    p.add_argument("--n_episodes", type=int, default=3,
                   help="Number of episodes to render per snapshot pair (default: 3).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for episode resets (default: 42).")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Use deterministic policy predictions (default: True).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class TagRenderer:
    """
    Pygame renderer for a single game state.
    Draws the grid, walls, tagger (red circle), and runner (blue circle),
    plus an info bar at the bottom with step count and game outcome.
    """

    def __init__(self, fps: int = 8):
        pygame.init()
        pygame.display.set_caption("Tag RL — Demo")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
        self.clock  = pygame.font.SysFont(None, 22)
        self.font   = pygame.font.SysFont("monospace", 16)
        self.fps_clock = pygame.time.Clock()
        self.fps    = fps

    def _layout(self):
        """Compute dynamic layout from current window size."""
        w, h = self.screen.get_size()
        margin     = max(4, min(MARGIN, w // 80))
        info_h     = INFO_HEIGHT
        cell_size  = max(8, min((w - 2 * margin) // GRID_SIZE,
                                (h - 2 * margin - info_h) // GRID_SIZE))
        grid_px    = cell_size * GRID_SIZE
        gx0        = (w - grid_px) // 2
        gy0        = (h - grid_px - info_h) // 2
        return w, h, cell_size, gx0, gy0, info_h

    def draw(self, gs: GridState, step: int, label: str = "") -> None:
        """Draw current game state."""
        self.screen.fill(BG_COLOR)

        w, h, cell_size, gx0, gy0, info_h = self._layout()

        # Draw grid cells
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x = gx0 + c * cell_size
                y = gy0 + r * cell_size
                color = WALL_COLOR if gs.walls[r, c] else CELL_COLOR
                pygame.draw.rect(self.screen, color, (x, y, cell_size, cell_size))
                pygame.draw.rect(self.screen, GRID_COLOR, (x, y, cell_size, cell_size), 1)

        # Draw tagger — purple during grace period (frozen), red when active
        tr, tc = int(gs.tagger_pos[0]), int(gs.tagger_pos[1])
        tx = gx0 + tc * cell_size + cell_size // 2
        ty = gy0 + tr * cell_size + cell_size // 2
        grace_active = gs.grace_steps_remaining > 0
        tagger_draw_color = TAGGER_GRACE_COLOR if grace_active else TAGGER_COLOR
        radius = max(4, cell_size // 2 - 4)
        pygame.draw.circle(self.screen, tagger_draw_color, (tx, ty), radius)
        if grace_active:
            # White outline ring indicates the tagger is frozen
            pygame.draw.circle(self.screen, (255, 255, 255), (tx, ty), radius + 3, width=2)
        # Label: T
        label_surf = self.font.render("T", True, (255, 255, 255))
        self.screen.blit(label_surf, label_surf.get_rect(center=(tx, ty)))

        # Draw runner (filled blue circle)
        rr, rc = int(gs.runner_pos[0]), int(gs.runner_pos[1])
        rx = gx0 + rc * cell_size + cell_size // 2
        ry = gy0 + rr * cell_size + cell_size // 2
        pygame.draw.circle(self.screen, RUNNER_COLOR, (rx, ry), max(4, cell_size // 2 - 4))
        # Label: R
        label_surf = self.font.render("R", True, (255, 255, 255))
        self.screen.blit(label_surf, label_surf.get_rect(center=(rx, ry)))

        # Info bar
        info_y = h - info_h
        pygame.draw.rect(self.screen, INFO_BG, (0, info_y, w, info_h))
        info_text = f"Step: {step:3d}/{MAX_STEPS}  Tags: {gs.total_tags}"
        if gs.grace_steps_remaining > 0:
            info_text += f"  [GRACE: {gs.grace_steps_remaining}]"
        if label:
            info_text += f"   {label}"
        text_surf = self.font.render(info_text, True, TEXT_COLOR)
        self.screen.blit(text_surf, (MARGIN, info_y + 10))

        pygame.display.flip()

    def tick(self) -> bool:
        """Process events; return False if user closed the window."""
        self.fps_clock.tick(self.fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
            if event.type == pygame.VIDEORESIZE:
                # pygame 1.x: must recreate the surface on resize
                self.screen = pygame.display.set_mode(
                    event.size, pygame.RESIZABLE
                )
        return True

    def capture_frame(self) -> np.ndarray:
        """Capture current screen as an H×W×3 uint8 numpy array."""
        surface_array = pygame.surfarray.array3d(self.screen)
        # surfarray gives (W, H, 3); transpose to (H, W, 3) for imageio/PIL
        return surface_array.transpose(1, 0, 2)

    def quit(self) -> None:
        pygame.quit()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def render_episode(
    tagger_model,
    runner_model,
    renderer: TagRenderer,
    seed: int,
    deterministic: bool,
    label: str = "",
    record: bool = False,
) -> list:
    """
    Run and render one episode.  Returns a list of frames (np.ndarray) if
    record=True, else an empty list.

    The game state is driven directly through GridState so we can query both
    agents' observations independently without routing through a Gym env.
    """
    from stable_baselines3 import PPO  # local import — never at module level in training

    gs = GridState(seed=seed)
    gs.reset()

    frames = []
    done   = False
    step   = 0

    while not done:
        # Handle window close / ESC
        if not renderer.tick():
            return frames

        # Draw current state
        renderer.draw(gs, step, label=label)

        if record:
            frames.append(renderer.capture_frame())

        if gs.done:
            # Show final frame for a moment
            import time; time.sleep(0.4)
            break

        # Get actions
        t_obs = gs.get_tagger_obs()
        r_obs = gs.get_runner_obs()

        if tagger_model is not None:
            t_action, _ = tagger_model.predict(t_obs, deterministic=deterministic)
            t_action = int(t_action)
        else:
            import random; t_action = random.randint(0, 8)

        if runner_model is not None:
            r_action, _ = runner_model.predict(r_obs, deterministic=deterministic)
            r_action = int(r_action)
        else:
            import random; r_action = random.randint(0, 4)

        gs.step(t_action, r_action)
        step += 1

    return frames


# ---------------------------------------------------------------------------
# GIF/MP4 saving
# ---------------------------------------------------------------------------

def save_gif(frames: list, path: str, fps: int) -> None:
    """
    Save a list of frames as an animated GIF using imageio (preferred) or Pillow.
    """
    if not frames:
        print("  [warn] No frames to save.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    try:
        import imageio
        imageio.mimsave(path, frames, fps=fps, loop=0)
        print(f"  Saved GIF: {path}  ({len(frames)} frames @ {fps} fps)")
        return
    except ImportError:
        pass

    # Fallback: Pillow
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        duration_ms = int(1000 / fps)
        imgs[0].save(
            path,
            save_all=True,
            append_images=imgs[1:],
            loop=0,
            duration=duration_ms,
        )
        print(f"  Saved GIF (Pillow): {path}  ({len(frames)} frames @ {fps} fps)")
        return
    except ImportError:
        pass

    print("  [warn] Neither imageio nor Pillow is installed.  "
          "Cannot save GIF.  Run: pip install imageio  or  pip install Pillow")


# ---------------------------------------------------------------------------
# Snapshot discovery helper
# ---------------------------------------------------------------------------

def find_snapshot_at_fraction(snapshots_dir: str, prefix: str, fraction: float) -> str:
    """
    Find the snapshot whose cycle number is closest to `fraction` of the
    maximum cycle.  Used for --stages mode to auto-select early/mid/late.
    """
    import glob as _glob
    pattern = os.path.join(snapshots_dir, f"{prefix}_*.zip")
    files   = sorted(_glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No {prefix} snapshots found in '{snapshots_dir}'. "
            f"Run agents/train.py first."
        )
    cycles = []
    for f in files:
        basename = os.path.basename(f)
        try:
            cycle = int(basename.replace(f"{prefix}_", "").replace(".zip", ""))
            cycles.append((cycle, f.replace(".zip", "")))
        except ValueError:
            pass
    cycles.sort()
    target = fraction * cycles[-1][0]
    closest = min(cycles, key=lambda x: abs(x[0] - target))
    return closest[1], closest[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    from stable_baselines3 import PPO
    from env.tag_env import TaggerEnv, RunnerEnv

    renderer = TagRenderer(fps=args.fps)

    try:
        if args.stages:
            # --- Three-stage mode ---
            fractions = [0.0, 0.5, 1.0]
            stage_labels = ["Early (random)", "Mid-training", "Late (converged)"]

            for frac, stage_label in zip(fractions, stage_labels):
                tagger_path, t_cycle = find_snapshot_at_fraction(
                    args.snapshots_dir, "tagger", frac
                )
                runner_path, r_cycle = find_snapshot_at_fraction(
                    args.snapshots_dir, "runner", frac
                )
                print(f"\nStage: {stage_label}  "
                      f"(tagger cycle {t_cycle}, runner cycle {r_cycle})")

                dummy_tagger_env = TaggerEnv(seed=0)
                dummy_runner_env = RunnerEnv(seed=0)
                tagger_model = PPO.load(tagger_path, env=dummy_tagger_env)
                runner_model = PPO.load(runner_path, env=dummy_runner_env)

                all_frames = []
                for ep in range(args.n_episodes):
                    frames = render_episode(
                        tagger_model, runner_model, renderer,
                        seed          = args.seed + ep,
                        deterministic = args.deterministic,
                        label         = f"{stage_label} | ep {ep+1}/{args.n_episodes}",
                        record        = args.save_dir is not None,
                    )
                    all_frames.extend(frames)

                if args.save_dir and all_frames:
                    stage_name = stage_label.split()[0].lower()
                    gif_path   = os.path.join(args.save_dir, f"{stage_name}_stage.gif")
                    save_gif(all_frames, gif_path, args.fps)

        elif args.replay:
            # --- Replay-all mode ---
            # Loads every snapshot in cycle order and plays one episode each.
            # Use --fps to control playback speed without retraining.
            import glob as _glob

            def _parse_snaps(prefix):
                files = sorted(_glob.glob(
                    os.path.join(args.snapshots_dir, f"{prefix}_*.zip")
                ))
                out = {}
                for f in files:
                    base = os.path.basename(f)
                    try:
                        cycle = int(base.replace(f"{prefix}_", "").replace(".zip", ""))
                        out[cycle] = f.replace(".zip", "")
                    except ValueError:
                        pass
                return out

            t_snaps = _parse_snaps("tagger")
            r_snaps = _parse_snaps("runner")
            common  = sorted(set(t_snaps) & set(r_snaps))

            if not common:
                print("[replay] No snapshot pairs found in "
                      f"'{args.snapshots_dir}'. Run agents/train.py first.")
            else:
                dummy_t = TaggerEnv(seed=0)
                dummy_r = RunnerEnv(seed=0)
                print(f"[replay] {len(common)} snapshots — "
                      f"cycles {common[0]}–{common[-1]}  |  "
                      f"fps={args.fps}  |  ESC or close window to skip ahead")

                for cycle in common:
                    print(f"  Cycle {cycle:4d} / {common[-1]}", end="\r", flush=True)
                    t_model = PPO.load(t_snaps[cycle], env=dummy_t)
                    r_model = PPO.load(r_snaps[cycle], env=dummy_r)
                    render_episode(
                        t_model, r_model, renderer,
                        seed          = args.seed + cycle,
                        deterministic = args.deterministic,
                        label         = f"Cycle {cycle} / {common[-1]}",
                        record        = False,
                    )
                print("\n[replay] Done.")

        else:
            # --- Single-episode mode ---
            if args.tagger is None or args.runner is None:
                print("Provide --tagger and --runner snapshot paths, "
                      "or use --stages for automatic three-stage rendering.")
                print("Example: python render/visualize.py "
                      "--tagger snapshots/tagger_0200 --runner snapshots/runner_0200")
                return

            dummy_tagger_env = TaggerEnv(seed=0)
            dummy_runner_env = RunnerEnv(seed=0)

            print(f"Loading tagger: {args.tagger}")
            tagger_model = PPO.load(args.tagger, env=dummy_tagger_env)

            print(f"Loading runner: {args.runner}")
            runner_model = PPO.load(args.runner, env=dummy_runner_env)

            all_frames = []
            for ep in range(args.n_episodes):
                print(f"Rendering episode {ep+1}/{args.n_episodes}...")
                frames = render_episode(
                    tagger_model, runner_model, renderer,
                    seed          = args.seed + ep,
                    deterministic = args.deterministic,
                    label         = f"ep {ep+1}/{args.n_episodes}",
                    record        = args.save is not None,
                )
                all_frames.extend(frames)

            if args.save and all_frames:
                save_gif(all_frames, args.save, args.fps)

    finally:
        renderer.quit()


if __name__ == "__main__":
    main()
