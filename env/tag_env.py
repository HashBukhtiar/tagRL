"""
env/tag_env.py — Custom Gymnasium environments for the multi-agent tag game.

Architecture: two separate Gymnasium envs (TaggerEnv, RunnerEnv) that each wrap
a shared GridState object. SB3 drives one env at a time; the frozen opponent's
action is computed inside step() so SB3 only sees a standard single-agent MDP.

--- Report-relevant design decisions ---

1. Two-env / shared-state architecture
   Each env is a standard Gymnasium env, so SB3 can plug in without modification.
   The GridState object holds ground-truth positions and is owned by the env that
   is currently being trained; the other env re-uses that same object during eval.
   This avoids synchronisation bugs that arise when two separate envs would drift.

2. Flat observation vector (MlpPolicy, not CNN)
   We encode all task-relevant information explicitly:
     Tagger (16-dim): [own_row_n, own_col_n, lk_opp_row_n, lk_opp_col_n, own_dr, own_dc,
      can_move_up, can_move_down, can_move_left, can_move_right,
      can_move_ul, can_move_ur, can_move_dl, can_move_dr, opp_visible, grace_n]
     Runner (12-dim): [own_row_n, own_col_n, lk_opp_row_n, lk_opp_col_n, own_dr, own_dc,
      can_move_up, can_move_down, can_move_left, can_move_right, opp_visible, grace_n]
   A CNN over raw 12×12 pixels would need many more samples to learn positional
   features from scratch, and the grid is small enough that explicit coordinates
   are unambiguous. Flat vector → MlpPolicy → faster training convergence.
   Movement flags directly encode whether a cardinal action would be blocked by a wall,
   so the agent learns navigation constraints without trial-and-error wall collisions.

3. Partial observability via line-of-sight (LOS) occlusion with last-known position
   If a wall cell lies on the ray between the two agents, the opponent's *current*
   position is not known. The observation reports the last-known opponent position
   (the most recent step where LOS was clear) plus an explicit binary `opp_visible`
   flag (1.0 = currently visible, 0.0 = using stale last-known position). Before the
   opponent is ever seen, the last-known position is −1 (a sentinel for "never seen").
   This gives the policy a principled search target when the opponent hides, rather
   than a useless −1 that conveys no spatial information. The explicit visibility flag
   lets the policy learn to distinguish "I see them now" from "I last saw them there".
   LOS is computed with Bresenham's line algorithm using integer arithmetic —
   conservative on diagonal corner touches.

4. Velocity (last displacement) in observation
   own_dr, own_dc ∈ {−1, 0, 1} encode the direction the agent moved last step.
   This gives the policy a one-step momentum signal without requiring recurrence.
   Without it, the policy cannot distinguish "I'm moving towards the opponent"
   from "I just bounced off a wall and am now stationary".

5. Reward design — shaped rewards for both agents for dense learning signals
   Tagger: +10 on catch, −0.1/step time penalty, plus three shaping components:
     (a) Potential-based distance shaping: F = γ·Φ(s') − Φ(s), Φ(s) = −dist.
         Provides a dense gradient toward the runner each step without altering
         the optimal policy (Ng et al., 1999).
     (b) STAY action penalty (−0.5): breaks standstill equilibria where the
         tagger avoids negative expected reward by doing nothing. Not applied
         during the grace period (forced STAY is not voluntary).
     (c) Revisit penalty (−0.2): penalises returning to a recently-visited cell,
         breaking 2-step loop traps common in early self-play training.
   Runner: +1/step (survival), −10 on catch, plus two shaping components:
     (a) Potential-based escape shaping: F = γ·dist_after − dist_before.
         Mirror of the tagger's shaping — dense gradient to maximise distance
         from the tagger each step without altering the optimal policy.
     (b) STAY action penalty (−0.1): discourages standing still when the tagger
         approaches, which is the dominant failure mode without this signal.

6. Simultaneous movement
   Both agents move at the same time each step. Sequential movement would give a
   first-mover advantage (the first-moving agent sees the updated position of the
   other and can react). Simultaneous movement is fairer and closer to real pursuit.

7. True tag with role-switching
   Episodes run for MAX_STEPS=500 without terminating on a catch. When the tagger
   catches the runner, roles swap: the old runner becomes the new tagger and is
   frozen (STAY forced) for GRACE_PERIOD=10 steps, giving the new runner a head
   start. The grace_n observation feature (grace_steps_remaining / GRACE_PERIOD)
   lets both agents reason about the freeze duration. The evaluation metric shifts
   from episode duration to total_tags per episode.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE    = 20   # 20×20 grid; border cells are walls; interior is 18×18
MAX_STEPS    = 500  # episode length cap; episodes always run to MAX_STEPS
GRACE_PERIOD = 10   # steps the new tagger is fully frozen after a role swap

N_ACTIONS = 5
OBS_DIM   = 12   # 6 scalars (own pos, lk opp pos, velocity) + 4 movement flags + 1 visibility flag + 1 grace flag

TAGGER_N_ACTIONS = 9    # 8 directions (4 cardinal + 4 diagonal) + STAY
TAGGER_OBS_DIM   = 16   # 6 scalars + 8 movement flags + 1 visibility flag + 1 grace flag

# Tagger reward shaping constants
SHAPING_GAMMA   = 0.99   # must match PPO gamma
SHAPING_SCALE   = 0.5    # weight on potential-based distance shaping
STAY_PENALTY    = -0.5   # extra penalty for tagger choosing STAY (action 4)
REVISIT_PENALTY = -0.2   # penalty for tagger returning to a recently-visited cell
REVISIT_WINDOW  = 8      # number of recent tagger positions tracked for loop detection

# Runner reward shaping constants
RUNNER_STAY_PENALTY = -0.1  # penalty for runner choosing STAY (discourage standing still)

# Interior obstacles: ~35 cells.
# Creates 4 corner hideouts (L-blocks), 2 horizontal chokepoints, and 1 vertical bar.
INTERIOR_OBSTACLES = [
    # Upper right corner room
    (1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (5, 11), (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17),
    (2, 12), (3, 12), (3, 13), (3, 14), (3, 15), (3, 16), (3, 17), (3, 18),
    (1, 17), (2, 9),

    # Upper left corner
    (2, 2), (2, 3), (2, 4), (3, 2), (4, 2), (5, 2), (5, 3), (5, 4),

    # Bottom bar
    (17, 2), (17, 3), (17, 4), (17, 5), (17, 7), (17, 8), (17, 9), (17, 10), (17, 11), (17, 12), (17, 13), (17, 14), (17, 15), (17, 16), (17, 17),

    # Middle islands
    (13, 3), (12, 3), (12, 4), (11, 4), (12, 5), (12, 6), (11, 6), (13, 6),
    (8, 9), (9, 9), (10, 9), (8, 10), (9, 10), (10, 10), (8, 11), (9, 11), (10, 11), (8, 12), (9, 12), (10, 12), (8, 13), (9, 13), (10, 13),

    # Middle blocks
    (14, 8), (14, 9), (15, 8),
    (13, 11),
    (15, 13),
    (13, 15), (14, 16),
]

# Action index → (delta_row, delta_col)
# Row increases downward; UP means row−1, DOWN means row+1.
ACTIONS = {
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
    4: ( 0,  0),   # STAY
    5: (-1, -1),   # UP-LEFT   (tagger only)
    6: (-1,  1),   # UP-RIGHT  (tagger only)
    7: ( 1, -1),   # DOWN-LEFT (tagger only)
    8: ( 1,  1),   # DOWN-RIGHT (tagger only)
}


# ---------------------------------------------------------------------------
# Shared grid state
# ---------------------------------------------------------------------------

class GridState:
    """
    Ground-truth game state shared between TaggerEnv and RunnerEnv.

    Owns the static wall map and the mutable agent positions. Both env classes
    hold a reference to the same GridState so their views are always consistent.
    Keeping state here (rather than duplicating it in each env) is the canonical
    way to implement shared-state multi-agent envs without a full MARL framework.

    The environment is role-centric, not agent-centric. tagger_pos and runner_pos
    are roles: when a tag occurs, these arrays are swapped so the new tagger is
    always at tagger_pos regardless of which physical agent it was before.
    """

    def __init__(self, grid_size: int = GRID_SIZE, seed: int = None):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)

        # Static wall map: 1 = wall, 0 = passable.
        # Border walls + interior obstacles for line-of-sight occlusion and hiding.
        self.walls = np.zeros((grid_size, grid_size), dtype=np.int8)
        self.walls[0, :]  = 1   # top border
        self.walls[-1, :] = 1   # bottom border
        self.walls[:, 0]  = 1   # left border
        self.walls[:, -1] = 1   # right border
        for r, c in INTERIOR_OBSTACLES:
            self.walls[r, c] = 1

        # Agent positions — initialised properly by reset()
        self.tagger_pos    = np.array([1, 1], dtype=np.int32)
        self.runner_pos    = np.array([grid_size - 2, grid_size - 2], dtype=np.int32)
        self.tagger_last_d = np.zeros(2, dtype=np.int32)
        self.runner_last_d = np.zeros(2, dtype=np.int32)

        # Last-known opponent positions (updated only when LOS is clear).
        # −1 = never seen yet (episode start sentinel, outside [0,1] range).
        self.tagger_last_known_runner = np.array([-1.0, -1.0], dtype=np.float32)
        self.runner_last_known_tagger = np.array([-1.0, -1.0], dtype=np.float32)

        self.step_count            = 0
        self.done                  = False
        self.grace_steps_remaining = 0   # steps the new tagger is frozen; 0 = free to move
        self.total_tags            = 0   # cumulative tag events this episode

        self._tagger_pos_history: deque = deque(maxlen=REVISIT_WINDOW)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _passable_cells(self):
        """List of all (row, col) positions that are not walls."""
        rows, cols = np.where(self.walls == 0)
        return list(zip(rows.tolist(), cols.tolist()))

    def _has_los(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        Bresenham line-of-sight: returns True if no wall cell lies on the
        integer rasterisation of the ray from a to b (exclusive of endpoint b,
        inclusive of start a). Both positions are assumed to be passable.
        """
        r0, c0 = int(a[0]), int(a[1])
        r1, c1 = int(b[0]), int(b[1])
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        r, c = r0, c0
        n = 1 + dr + dc           # total cells visited (including start)
        r_inc = 1 if r1 > r0 else -1
        c_inc = 1 if c1 > c0 else -1
        error = dr - dc
        dr *= 2
        dc *= 2
        for _ in range(n - 1):    # iterate n-1 steps, stopping before endpoint
            if self.walls[r, c]:  # wall at current cell blocks sight
                return False
            if error > 0:
                r += r_inc
                error -= dc
            else:
                c += c_inc
                error += dr
        return True

    def _get_movement_flags(self, pos: np.ndarray, n_dirs: int = 4) -> np.ndarray:
        """
        Return a float32 array of length n_dirs indicating passability.
        n_dirs=4 → cardinal directions only (UP, DOWN, LEFT, RIGHT).
        n_dirs=8 → cardinals plus diagonals (UP-LEFT, UP-RIGHT, DOWN-LEFT, DOWN-RIGHT).
        Each value is 1.0 (passable) or 0.0 (wall blocks movement).
        """
        flags = []
        for action in range(n_dirs):
            dr, dc = ACTIONS[action]
            candidate = pos + np.array([dr, dc], dtype=np.int32)
            candidate = np.clip(candidate, 0, self.grid_size - 1)
            can_move = 0.0 if self.walls[candidate[0], candidate[1]] else 1.0
            flags.append(can_move)
        return np.array(flags, dtype=np.float32)

    def _try_move(self, pos: np.ndarray, action: int):
        """
        Attempt to move from pos using action. Returns (new_pos, actual_delta).
        Movement into a wall is blocked; agent stays in place and delta is (0,0).
        """
        dr, dc = ACTIONS[action]
        candidate = pos + np.array([dr, dc], dtype=np.int32)
        candidate = np.clip(candidate, 0, self.grid_size - 1)
        if self.walls[candidate[0], candidate[1]]:
            return pos.copy(), np.zeros(2, dtype=np.int32)
        return candidate, np.array([dr, dc], dtype=np.int32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """
        Randomise agent starting positions (non-overlapping), reset counters.
        Both agents are placed uniformly at random among all passable cells.
        """
        passable = self._passable_cells()
        idx = self.rng.choice(len(passable), size=2, replace=False)
        self.tagger_pos    = np.array(passable[idx[0]], dtype=np.int32)
        self.runner_pos    = np.array(passable[idx[1]], dtype=np.int32)
        self.tagger_last_d = np.zeros(2, dtype=np.int32)
        self.runner_last_d = np.zeros(2, dtype=np.int32)
        self.tagger_last_known_runner = np.array([-1.0, -1.0], dtype=np.float32)
        self.runner_last_known_tagger = np.array([-1.0, -1.0], dtype=np.float32)
        self.step_count            = 0
        self.done                  = False
        self.grace_steps_remaining = 0
        self.total_tags            = 0
        self._tagger_pos_history.clear()

    def step(self, tagger_action: int, runner_action: int):
        """
        Apply both actions simultaneously, update positions, handle role-switching.
        Returns (tagger_reward, runner_reward, terminated, truncated, info).

        Episodes never terminate on a catch — roles swap instead. The only terminal
        signal is truncation at MAX_STEPS=500 steps.

        Grace period: for GRACE_PERIOD steps after a role swap, the new tagger's
        submitted action is overridden to STAY (fully frozen). The STAY_PENALTY is
        not applied during this forced freeze.

        Tagger reward = −0.1 (time penalty)
                      + γ·Φ(s') − Φ(s)  where Φ(s) = −dist  (potential-based shaping)
                      + STAY_PENALTY if voluntary STAY (not during grace period)
                      + REVISIT_PENALTY if new cell was recently visited  (break loops)
                      + 10.0 on catch (role swap event)
        Runner reward = +1.0 (survival)
                      + γ·dist_after − dist_before  (potential-based escape shaping)
                      + RUNNER_STAY_PENALTY if action == STAY  (discourage standing still)
                      − 10.0 on catch (role swap event)
        """
        assert not self.done, "step() called on a finished episode — call reset() first"

        # Grace period: new tagger cannot move. Record whether this step is forced
        # before overriding the action, so we don't apply STAY_PENALTY for it.
        grace_forced = self.grace_steps_remaining > 0
        if grace_forced:
            tagger_action = 4   # force STAY

        dist_before = float(np.sum(np.abs(self.tagger_pos - self.runner_pos)))

        # === BOTH AGENTS MOVE SIMULTANEOUSLY ===
        new_tagger, t_delta = self._try_move(self.tagger_pos, tagger_action)
        self.tagger_pos    = new_tagger
        self.tagger_last_d = t_delta

        new_runner, r_delta = self._try_move(self.runner_pos, runner_action)
        self.runner_pos    = new_runner
        self.runner_last_d = r_delta

        self.step_count += 1

        # Base per-step rewards
        tagger_reward = -0.1   # time penalty: tagger must actively pursue
        runner_reward =  1.0   # survival bonus: runner wants to last as long as possible

        # Potential-based distance shaping — dense chase/escape signals.
        # F(s, s') = γ·Φ(s') − Φ(s), Φ(s) = −dist preserves the optimal policy.
        dist_after = float(np.sum(np.abs(self.tagger_pos - self.runner_pos)))
        tagger_reward += (SHAPING_GAMMA * (-dist_after) - (-dist_before)) * SHAPING_SCALE
        # Runner shaping: reward increasing distance from tagger (Φ(s) = dist).
        runner_reward += (SHAPING_GAMMA * dist_after - dist_before) * SHAPING_SCALE

        # Extra STAY penalties — discourages standstill equilibria.
        # Tagger STAY_PENALTY only applies when the stay was voluntary (not grace-forced).
        if tagger_action == 4 and not grace_forced:
            tagger_reward += STAY_PENALTY
        if runner_action == 4:
            runner_reward += RUNNER_STAY_PENALTY

        # Revisit penalty — discourages 2-step loops.
        tagger_pos_key = (int(self.tagger_pos[0]), int(self.tagger_pos[1]))
        if tagger_pos_key in self._tagger_pos_history:
            tagger_reward += REVISIT_PENALTY
        self._tagger_pos_history.append(tagger_pos_key)

        # === TAG DETECTION AND ROLE SWAP ===
        # Catch is checked once after both agents have moved.
        tagged = bool(np.array_equal(self.tagger_pos, self.runner_pos))
        if tagged:
            self.total_tags += 1
            tagger_reward   += 10.0
            runner_reward   += -10.0

            # Swap role-centric position and velocity arrays.
            # tagger_pos/runner_pos are roles, not identities — swapping them
            # makes the old runner the new tagger without any identity tracking.
            self.tagger_pos,   self.runner_pos   = self.runner_pos.copy(),   self.tagger_pos.copy()
            self.tagger_last_d, self.runner_last_d = self.runner_last_d.copy(), self.tagger_last_d.copy()

            # Swap last-known positions: each agent's stored knowledge flips with the role.
            self.tagger_last_known_runner, self.runner_last_known_tagger = (
                self.runner_last_known_tagger.copy(),
                self.tagger_last_known_runner.copy(),
            )

            # Freeze the new tagger and clear its revisit history (fresh start).
            self.grace_steps_remaining = GRACE_PERIOD
            self._tagger_pos_history.clear()

        # Grace period countdown (happens after the tag check so GRACE_PERIOD is the
        # full number of frozen steps, not GRACE_PERIOD-1).
        if self.grace_steps_remaining > 0:
            self.grace_steps_remaining -= 1

        # === TERMINATION ===
        # Episodes never end on a catch — only on MAX_STEPS.
        timed_out  = self.step_count >= MAX_STEPS
        terminated = False   # always False; catch no longer ends the episode
        truncated  = timed_out
        if truncated:
            self.done = True

        info = {
            "tagged":                  tagged,
            "total_tags":              self.total_tags,
            "grace_steps_remaining":   self.grace_steps_remaining,
            "timed_out":               timed_out,
            "step_count":              self.step_count,
        }
        return tagger_reward, runner_reward, terminated, truncated, info

    def get_tagger_obs(self) -> np.ndarray:
        """
        Tagger's observation vector (float32, shape (TAGGER_OBS_DIM,) = (16,)):
          [own_row_n, own_col_n, lk_runner_row_n, lk_runner_col_n, own_dr, own_dc,
           can_move_up, can_move_down, can_move_left, can_move_right,
           can_move_ul, can_move_ur, can_move_dl, can_move_dr, opp_visible, grace_n]

        Positions are normalised to [0, 1] by dividing by (grid_size − 1).
        Velocity components are in {−1, 0, 1} (raw deltas, already small).
        lk_runner_{row,col}_n is the *last-known* runner position: updated each step
        when LOS is clear, held fixed when LOS is blocked. Before the runner has ever
        been seen it is −1 (episode-start sentinel). opp_visible is 1.0 when the runner
        is currently visible (LOS clear), 0.0 when using the stale last-known position.
        grace_n = grace_steps_remaining / GRACE_PERIOD ∈ [0, 1]: how long the tagger
        is still frozen (0 = free to move). Lets the policy know when it will unfreeze.
        Movement flags are 1.0 (passable) or 0.0 (blocked by wall) for each direction.
        """
        norm = float(self.grid_size - 1)
        own_r = self.tagger_pos[0] / norm
        own_c = self.tagger_pos[1] / norm

        if self._has_los(self.tagger_pos, self.runner_pos):
            lk_r = self.runner_pos[0] / norm
            lk_c = self.runner_pos[1] / norm
            self.tagger_last_known_runner[:] = [lk_r, lk_c]
            visible = 1.0
        else:
            lk_r, lk_c = float(self.tagger_last_known_runner[0]), float(self.tagger_last_known_runner[1])
            visible = 0.0

        dr = float(self.tagger_last_d[0])
        dc = float(self.tagger_last_d[1])
        move_flags = self._get_movement_flags(self.tagger_pos, n_dirs=8)
        grace_n = float(self.grace_steps_remaining) / float(GRACE_PERIOD)
        return np.concatenate([[own_r, own_c, lk_r, lk_c, dr, dc], move_flags, [visible, grace_n]]).astype(np.float32)

    def get_runner_obs(self) -> np.ndarray:
        """
        Runner's observation vector (float32, shape (OBS_DIM,) = (12,)):
          [own_row_n, own_col_n, lk_tagger_row_n, lk_tagger_col_n, own_dr, own_dc,
           can_move_up, can_move_down, can_move_left, can_move_right, opp_visible, grace_n]

        Same partial observability logic as the tagger: the last-known tagger position
        is reported when LOS is blocked, together with an opp_visible flag (1.0 = tagger
        currently in sight, 0.0 = using stale last-known position). Before the tagger
        has ever been seen, the last-known position is −1 (episode-start sentinel).
        grace_n = grace_steps_remaining / GRACE_PERIOD ∈ [0, 1]: when > 0, the runner
        knows the tagger is frozen and can plan an escape. Lets the runner exploit the
        grace window to maximise separation distance before the tagger unfreezes.
        Movement flags are 1.0 (passable) or 0.0 (blocked by wall) for each cardinal
        direction, allowing the agent to learn immediate navigation constraints.
        """
        norm = float(self.grid_size - 1)
        own_r = self.runner_pos[0] / norm
        own_c = self.runner_pos[1] / norm

        if self._has_los(self.runner_pos, self.tagger_pos):
            lk_r = self.tagger_pos[0] / norm
            lk_c = self.tagger_pos[1] / norm
            self.runner_last_known_tagger[:] = [lk_r, lk_c]
            visible = 1.0
        else:
            lk_r, lk_c = float(self.runner_last_known_tagger[0]), float(self.runner_last_known_tagger[1])
            visible = 0.0

        dr = float(self.runner_last_d[0])
        dc = float(self.runner_last_d[1])
        move_flags = self._get_movement_flags(self.runner_pos)
        grace_n = float(self.grace_steps_remaining) / float(GRACE_PERIOD)
        return np.concatenate([[own_r, own_c, lk_r, lk_c, dr, dc], move_flags, [visible, grace_n]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Gymnasium environments
# ---------------------------------------------------------------------------

class TaggerEnv(gym.Env):
    """
    Single-agent Gymnasium env for the Tagger.

    From SB3's perspective this is a standard MDP: the tagger picks an action,
    the env returns (obs, reward, terminated, truncated, info).  Internally the
    env also computes the runner's action from the frozen opponent model and
    applies it to the shared GridState.

    set_opponent(model) is called before each training cycle to swap in the
    latest frozen runner snapshot.  If no opponent is set, the runner acts
    randomly — useful for the very first training cycle.
    """

    metadata = {"render_modes": []}

    def __init__(self, grid_size: int = GRID_SIZE, seed: int = None):
        super().__init__()
        self.grid_state     = GridState(grid_size=grid_size, seed=seed)
        self.opponent_model = None   # frozen RunnerModel; None → uniform random

        self.action_space = spaces.Discrete(TAGGER_N_ACTIONS)

        # Bounds: normalised positions ∈ [0,1], masked to −1; velocity ∈ {−1,0,1};
        # wall patch ∈ {0,1}; grace_n ∈ [0,1].  Using [−1, 1] as a universal safe bound.
        low  = np.full(TAGGER_OBS_DIM, -1.0, dtype=np.float32)
        high = np.full(TAGGER_OBS_DIM,  1.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def set_opponent(self, model) -> None:
        """Swap in a new frozen runner model.  Called between training cycles."""
        self.opponent_model = model

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.grid_state.rng = np.random.default_rng(seed)
        self.grid_state.reset()
        return self.grid_state.get_tagger_obs(), {}

    def step(self, tagger_action: int):
        # Compute the runner's action from the frozen opponent model.
        runner_obs = self.grid_state.get_runner_obs()
        if self.opponent_model is not None:
            runner_action, _ = self.opponent_model.predict(runner_obs, deterministic=False)
            runner_action = int(runner_action)
        else:
            runner_action = self.grid_state.rng.integers(0, N_ACTIONS)

        tagger_reward, _, terminated, truncated, info = self.grid_state.step(
            int(tagger_action), int(runner_action)
        )

        obs = self.grid_state.get_tagger_obs()
        return obs, float(tagger_reward), terminated, truncated, info


class RunnerEnv(gym.Env):
    """
    Single-agent Gymnasium env for the Runner.

    Mirror of TaggerEnv: the tagger's action is computed from the frozen
    opponent model inside step().  From SB3's perspective, a standard MDP.
    """

    metadata = {"render_modes": []}

    def __init__(self, grid_size: int = GRID_SIZE, seed: int = None):
        super().__init__()
        self.grid_state     = GridState(grid_size=grid_size, seed=seed)
        self.opponent_model = None   # frozen TaggerModel; None → uniform random

        self.action_space = spaces.Discrete(N_ACTIONS)

        low  = np.full(OBS_DIM, -1.0, dtype=np.float32)
        high = np.full(OBS_DIM,  1.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def set_opponent(self, model) -> None:
        """Swap in a new frozen tagger model.  Called between training cycles."""
        self.opponent_model = model

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.grid_state.rng = np.random.default_rng(seed)
        self.grid_state.reset()
        return self.grid_state.get_runner_obs(), {}

    def step(self, runner_action: int):
        # Compute the tagger's action from the frozen opponent model.
        tagger_obs = self.grid_state.get_tagger_obs()
        if self.opponent_model is not None:
            tagger_action, _ = self.opponent_model.predict(tagger_obs, deterministic=False)
            tagger_action = int(tagger_action)
        else:
            tagger_action = self.grid_state.rng.integers(0, TAGGER_N_ACTIONS)

        _, runner_reward, terminated, truncated, info = self.grid_state.step(
            int(tagger_action), int(runner_action)
        )

        obs = self.grid_state.get_runner_obs()
        return obs, float(runner_reward), terminated, truncated, info


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("TaggerEnv sanity check")
    print("=" * 60)

    tagger_env = TaggerEnv(seed=42)
    obs, info = tagger_env.reset()
    print(f"obs shape : {obs.shape}  dtype: {obs.dtype}")
    print(f"obs[:6]   : {obs[:6]}  (own_r, own_c, lk_opp_r, lk_opp_c, dr, dc)")
    print(f"obs[6:14] : {obs[6:14]}  (movement flags: up, down, left, right, ul, ur, dl, dr)")
    print(f"obs[14]   : {obs[14]}  (opp_visible)")
    print(f"obs[15]   : {obs[15]}  (grace_n)")
    assert obs.shape == (TAGGER_OBS_DIM,), f"Expected ({TAGGER_OBS_DIM},), got {obs.shape}"

    total_r = 0.0
    tag_count = 0
    for step_i in range(1, MAX_STEPS + 2):
        action = tagger_env.action_space.sample()
        obs, reward, terminated, truncated, info = tagger_env.step(action)
        total_r += reward
        if info["tagged"]:
            tag_count += 1
        assert not terminated, f"terminated=True at step {step_i} — should never happen"
        print(f"  step {step_i:3d}: action={action}  reward={reward:+.2f}  "
              f"terminated={terminated}  truncated={truncated}  "
              f"tags={info['total_tags']}  grace={info['grace_steps_remaining']}  info={info}")
        if terminated or truncated:
            break
    print(f"Episode ended at step {info['step_count']}.  total_tags={info['total_tags']}")
    print(f"Total tagger reward: {total_r:.2f}\n")

    print("=" * 60)
    print("RunnerEnv sanity check")
    print("=" * 60)

    runner_env = RunnerEnv(seed=0)
    obs, info = runner_env.reset()
    print(f"obs shape : {obs.shape}  dtype: {obs.dtype}")
    print(f"obs[:6]   : {obs[:6]}  (own_r, own_c, lk_opp_r, lk_opp_c, dr, dc)")
    print(f"obs[6:10] : {obs[6:10]}  (movement flags: up, down, left, right)")
    print(f"obs[10]   : {obs[10]}  (opp_visible)")
    print(f"obs[11]   : {obs[11]}  (grace_n)")
    assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"

    total_r = 0.0
    for step_i in range(1, MAX_STEPS + 2):
        action = runner_env.action_space.sample()
        obs, reward, terminated, truncated, info = runner_env.step(action)
        total_r += reward
        assert not terminated, f"terminated=True at step {step_i} — should never happen"
        print(f"  step {step_i:3d}: action={action}  reward={reward:+.2f}  "
              f"terminated={terminated}  truncated={truncated}  "
              f"tags={info['total_tags']}  grace={info['grace_steps_remaining']}")
        if terminated or truncated:
            break
    print(f"Episode ended at step {info['step_count']}.  total_tags={info['total_tags']}")
    print(f"Total runner reward: {total_r:.2f}")

    print("\n[PASS] obs shapes correct, terminated always False, "
          "episodes truncate at MAX_STEPS, total_tags tracked.")
    sys.exit(0)
