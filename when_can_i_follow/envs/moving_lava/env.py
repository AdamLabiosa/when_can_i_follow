import heapq
from collections import deque

import gymnasium as gym
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv

# Direction vectors indexed by MiniGrid dir encoding: right, down, left, up
_DIR_VECS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# Object types that are safe to overwrite with the path marker in the obs image
_OVERWRITABLE_TYPES = {OBJECT_TO_IDX["empty"], OBJECT_TO_IDX["floor"]}

# RGB color and alpha used to paint the path overlay onto rendered frames
_PATH_COLOR_RGB = np.array([30, 144, 255], dtype=np.float32)  # dodger blue
_PATH_ALPHA = 0.45

# State-channel values that encode movement direction in path cells
# (1=right, 2=down, 3=left, 4=up; 0 = unknown/fallback)
_DIR_TO_STATE: dict[tuple[int, int], int] = {
    (1, 0): 1, (0, 1): 2, (-1, 0): 3, (0, -1): 4,
}

# Per-direction overlay colors for rendered frames (RGB float32)
_DIR_COLORS: dict[int, np.ndarray] = {
    0: np.array([30,  144, 255], dtype=np.float32),  # unknown  → blue
    1: np.array([30,  144, 255], dtype=np.float32),  # right    → blue
    2: np.array([50,  220,  80], dtype=np.float32),  # down     → green
    3: np.array([255, 210,  30], dtype=np.float32),  # left     → yellow
    4: np.array([255, 100,  30], dtype=np.float32),  # up       → orange
}


class movingLavaEnv(MiniGridEnv):
    """NxN empty grid with vertically-sliding lava lines.

    The agent starts in the middle of the left side facing right and must reach
    a randomly chosen cell on the right side.  Vertical strips of lava are
    scattered across the grid; each strip slides up or down by one cell every
    step and bounces off the top/bottom walls.

    Actions: 0=turn left, 1=turn right, 2=move forward, 3=request plan.

    Parameters
    ----------
    grid_size      : Full grid side length, including outer walls (so the
                     playable interior is (grid_size-2) x (grid_size-2)).
    min_lava_len   : Minimum vertical cell-count of a lava strip   (A).
    max_lava_len   : Maximum vertical cell-count of a lava strip   (B).
    min_x_gap      : Minimum column gap between consecutive strips  (C).
    max_x_gap      : Maximum column gap between consecutive strips  (D).
    min_y_gap      : Minimum vertical variation between strip starts (E).
    max_y_gap      : Maximum vertical variation between strip starts (F).
    lava_buffer    : Manhattan-distance buffer the planner keeps from lava.
    plan_delay     : Steps between a plan request and its delivery.
    frame_stack    : Number of consecutive frames to stack (1 = disabled).
    path_dir_colors: Encode movement direction via color in the path overlay.
    """

    def __init__(
        self,
        grid_size: int = 15,
        min_lava_len: int = 2,
        max_lava_len: int = 5,
        min_x_gap: int = 2,
        max_x_gap: int = 4,
        min_y_gap: int = 1,
        max_y_gap: int = 3,
        lava_buffer: int = 1,
        plan_delay: int = 3,
        frame_stack: int = 1,
        path_dir_colors: bool = True,
        **kwargs,
    ):
        self.grid_size = grid_size
        self.min_lava_len = min_lava_len
        self.max_lava_len = max_lava_len
        self.min_x_gap = min_x_gap
        self.max_x_gap = max_x_gap
        self.min_y_gap = min_y_gap
        self.max_y_gap = max_y_gap
        self.lava_buffer = lava_buffer
        self.plan_delay = plan_delay
        self._frame_stack = frame_stack
        self.path_dir_colors = path_dir_colors

        # Runtime state — initialised before super().__init__ so _gen_grid runs safely
        self._lava_lines: list[list] = []   # each entry: [x, y_top, length, direction]
        self.lava_cells: set[tuple[int, int]] = set()
        self._goal_pos: tuple[int, int] = (grid_size - 2, grid_size // 2)
        self.plan_started_at: int | None = None
        self.time = 0

        mission_space = MissionSpace(mission_func=lambda: "reach the green goal")
        super().__init__(
            mission_space=mission_space,
            width=grid_size,
            height=grid_size,
            max_steps=4 * grid_size * grid_size,
            **kwargs,
        )

        # Override to 4 discrete actions (0=left, 1=right, 2=forward, 3=plan)
        self.action_space = gym.spaces.Discrete(4)

        # Build observation space, optionally stacking frames
        single_img_shape = self.observation_space["image"].shape  # (H, W, C)
        self._frame_shape = single_img_shape
        if self._frame_stack > 1:
            self._frame_buffer: deque[np.ndarray] | None = deque(
                [np.zeros(single_img_shape, dtype=np.uint8)] * self._frame_stack,
                maxlen=self._frame_stack,
            )
            image_shape = (*single_img_shape[:2], single_img_shape[2] * self._frame_stack)
        else:
            self._frame_buffer = None
            image_shape = single_img_shape

        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
            # 1 while a plan is being computed (delay not yet elapsed)
            "plan_computing": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            # 1 once the completed plan has been delivered to the agent
            "plan_ready": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        self.plan: list[int] | None = None
        self.plan_path: list[tuple[int, int]] = []
        self.plan_request_pos: tuple[int, int] | None = None
        self.plan_request_dir: int | None = None

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Agent always starts in the middle of the left side, facing right
        agent_y = height // 2
        self.agent_pos = np.array([1, agent_y])
        self.agent_dir = 0  # right

        # Goal: random interior y on the right side
        goal_y = int(self.np_random.integers(1, height - 1))
        self.grid.set(width - 2, goal_y, Goal())
        self._goal_pos = (width - 2, goal_y)

        self.mission = "reach the green goal"

    # ------------------------------------------------------------------
    # Lava line management
    # ------------------------------------------------------------------

    def _generate_lava_lines(self) -> None:
        """Scatter vertical lava strips across the grid interior.

        Strips are generated left-to-right.  The x gap between consecutive
        strips is drawn from [min_x_gap, max_x_gap] and the y-center wanders
        by [min_y_gap, max_y_gap] each step so lines are spread vertically.
        Each strip is assigned a random up/down direction for this episode.
        """
        w = self.width
        h = self.height
        lines: list[list] = []

        # First strip x: at least min_x_gap from the left wall
        x = 1 + int(self.np_random.integers(self.min_x_gap, self.max_x_gap + 1))
        y_center = int(self.np_random.integers(1, h - 1))

        while x <= w - 3:  # leave at least one empty column before the right wall
            length = int(self.np_random.integers(self.min_lava_len, self.max_lava_len + 1))

            # Clamp length so the strip fits inside the interior rows [1, h-2]
            max_y_top = h - 1 - length      # y_top + length - 1 <= h - 2
            if max_y_top < 1:
                length = h - 2
                max_y_top = 1

            y_top = int(self.np_random.integers(1, max_y_top + 1))
            direction = 1 if self.np_random.random() < 0.5 else -1
            lines.append([x, y_top, length, direction])

            # Advance to the next strip position
            x_gap = int(self.np_random.integers(self.min_x_gap, self.max_x_gap + 1))
            y_gap = int(self.np_random.integers(self.min_y_gap, self.max_y_gap + 1))
            x += x_gap
            y_sign = 1 if self.np_random.random() < 0.5 else -1
            y_center = max(1, min(h - 2, y_center + y_sign * y_gap))

        self._lava_lines = lines

    def _update_lava_cells(self) -> None:
        """Sync `lava_cells` and the underlying grid with current `_lava_lines`."""
        # Erase old lava from the grid
        for x, y in self.lava_cells:
            cell = self.grid.get(x, y)
            if cell is not None and cell.type == "lava":
                self.grid.set(x, y, None)
        self.lava_cells = set()

        gx, gy = self._goal_pos
        for line in self._lava_lines:
            x, y_top, length, _ = line
            for dy in range(length):
                cy = y_top + dy
                if not (1 <= x <= self.width - 2 and 1 <= cy <= self.height - 2):
                    continue
                if (x, cy) == (gx, gy):
                    continue  # never overwrite the goal
                self.grid.set(x, cy, Lava())
                self.lava_cells.add((x, cy))

    def _move_lava(self) -> None:
        """Advance each strip one cell in its direction, bouncing at the walls."""
        h = self.height
        for line in self._lava_lines:
            x, y_top, length, direction = line
            new_y_top = y_top + direction

            # Clamp and reverse direction on boundary contact
            clamped = max(1, min(new_y_top, h - 1 - length))
            if clamped != new_y_top:
                line[3] = -direction  # bounce
            line[1] = clamped

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        self.time = 0
        self.plan_started_at = None
        self.lava_cells = set()
        self._lava_lines = []
        self.plan = None
        self.plan_path = []
        self.plan_request_pos = None
        self.plan_request_dir = None

        obs, info = super().reset(*args, **kwargs)

        self._generate_lava_lines()
        self._update_lava_cells()

        first_frame = obs["image"]
        if self._frame_stack > 1:
            self._frame_buffer = deque(
                [first_frame] * self._frame_stack,
                maxlen=self._frame_stack,
            )

        return self._make_obs(first_frame), info

    def step(self, action):
        self.time += 1

        # Move lava before processing the action
        self._move_lava()
        self._update_lava_cells()

        # If lava entered the agent's cell, end the episode immediately
        if tuple(self.agent_pos) in self.lava_cells:
            image_obs = self.gen_obs()["image"]
            return self._make_obs(image_obs), 0.0, True, False, {"lava_death": True}

        # Deliver a pending plan once the delay has elapsed
        if self.plan_started_at is not None and self.time - self.plan_started_at >= self.plan_delay:
            if self.plan is not None:
                self.plan_path = self._actions_to_world_coords(
                    self.plan_request_pos, self.plan_request_dir, self.plan
                )
            else:
                self.plan_path = []
            self.plan_started_at = None

        if action == 3:
            # Request a plan
            self.plan_started_at = self.time
            self.plan_request_pos = tuple(self.agent_pos)
            self.plan_request_dir = self.agent_dir
            self.plan = self.get_plan()

            image_obs = self.gen_obs()["image"]
            return self._make_obs(image_obs), 0.0, False, self.time >= self.max_steps, {}

        obs, reward, terminated, truncated, info = super().step(action)
        reward = 1.0 if terminated and reward > 0 else 0.0
        return self._make_obs(obs["image"]), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _make_obs(self, image: np.ndarray) -> dict:
        """Build the dict observation from a raw agent-POV image array."""
        if self._frame_buffer is not None:
            self._frame_buffer.append(image)
            obs_image = np.concatenate(list(self._frame_buffer), axis=-1)
        else:
            obs_image = image
        return {
            "image": obs_image,
            "plan_computing": np.array(
                [1.0 if self.plan_started_at is not None else 0.0], dtype=np.float32
            ),
            "plan_ready": np.array(
                [1.0 if self.plan_path else 0.0], dtype=np.float32
            ),
        }

    def gen_obs(self):
        obs = super().gen_obs()

        if self.plan_path:
            image = obs["image"].copy()
            dirs = self._plan_directions() if self.path_dir_colors else None
            for i, (wx, wy) in enumerate(self.plan_path):
                vx, vy = self.get_view_coords(wx, wy)
                if 0 <= vx < self.agent_view_size and 0 <= vy < self.agent_view_size:
                    if image[vx, vy, 0] in _OVERWRITABLE_TYPES:
                        d = dirs[i] if dirs is not None else 0
                        image[vx, vy] = np.array(
                            [OBJECT_TO_IDX["floor"], COLOR_TO_IDX["blue"], d],
                            dtype=np.uint8,
                        )
            obs["image"] = image

        return obs

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def get_full_render(self, highlight, tile_size):
        img = MiniGridEnv.get_full_render(self, highlight, tile_size)
        if self.plan_path:
            colors = (
                [_DIR_COLORS[d] for d in self._plan_directions()]
                if self.path_dir_colors
                else None
            )
            img = self._overlay_path(img, tile_size, self.plan_path, colors)
        return img

    def get_pov_render(self, tile_size):
        img = MiniGridEnv.get_pov_render(self, tile_size)
        if self.plan_path:
            dirs = self._plan_directions() if self.path_dir_colors else None
            visible: list[tuple[int, int]] = []
            visible_colors: list[np.ndarray] | None = [] if self.path_dir_colors else None
            for i, (wx, wy) in enumerate(self.plan_path):
                vx, vy = self.get_view_coords(wx, wy)
                if 0 <= vx < self.agent_view_size and 0 <= vy < self.agent_view_size:
                    visible.append((vx, vy))
                    if visible_colors is not None and dirs is not None:
                        visible_colors.append(_DIR_COLORS[dirs[i]])
            if visible:
                img = self._overlay_path(img, tile_size, visible, visible_colors)
        return img

    def _plan_directions(self) -> list[int]:
        """Return a direction-state int for every cell in plan_path."""
        path = self.plan_path
        out: list[int] = []
        for i, (wx, wy) in enumerate(path):
            if i + 1 < len(path):
                nx, ny = path[i + 1]
                d = _DIR_TO_STATE.get((nx - wx, ny - wy), 0)
            elif i > 0:
                px, py = path[i - 1]
                d = _DIR_TO_STATE.get((wx - px, wy - py), 0)
            else:
                d = 0
            out.append(d)
        return out

    def _overlay_path(
        self,
        img: np.ndarray,
        tile_size: int,
        cells: list[tuple[int, int]],
        cell_colors: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return a copy of `img` with a semi-transparent colored overlay on each cell."""
        img = img.copy()
        h, w = img.shape[:2]
        for i, (cx, cy) in enumerate(cells):
            color = cell_colors[i] if cell_colors is not None else _PATH_COLOR_RGB
            ymin, ymax = cy * tile_size, (cy + 1) * tile_size
            xmin, xmax = cx * tile_size, (cx + 1) * tile_size
            if ymin < 0 or xmin < 0 or ymax > h or xmax > w:
                continue
            patch = img[ymin:ymax, xmin:xmax].astype(np.float32)
            patch = patch * (1 - _PATH_ALPHA) + color * _PATH_ALPHA
            img[ymin:ymax, xmin:xmax] = patch.astype(np.uint8)
        return img

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _actions_to_world_coords(
        self,
        start_pos: tuple[int, int],
        start_dir: int,
        actions: list[int],
    ) -> list[tuple[int, int]]:
        """Replay a list of actions and return every world cell entered by a forward move."""
        x, y = start_pos
        d = start_dir
        coords: list[tuple[int, int]] = []
        for action in actions:
            if action == 0:      # turn left
                d = (d - 1) % 4
            elif action == 1:    # turn right
                d = (d + 1) % 4
            else:                # forward
                dx, dy = _DIR_VECS[d]
                x, y = x + dx, y + dy
                coords.append((x, y))
        return coords

    def _lava_buffer_cells(self, n: int) -> set[tuple[int, int]]:
        """Return all cells within Manhattan distance n of any lava cell."""
        if n <= 0 or not self.lava_cells:
            return set()
        buffer: set[tuple[int, int]] = set()
        for lx, ly in self.lava_cells:
            for dx in range(-n, n + 1):
                rem = n - abs(dx)
                for dy in range(-rem, rem + 1):
                    nx, ny = lx + dx, ly + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        buffer.add((nx, ny))
        return buffer

    def _astar(
        self,
        pos: tuple,
        agent_dir: int,
        output: str,
        extra_blocked: set[tuple[int, int]],
    ):
        """Core A* over the (x, y, dir) state space.

        extra_blocked: cells treated as unwalkable in addition to actual lava.
        Returns the plan in the requested output format, or None if unreachable.
        """
        gx, gy = self._goal_pos

        def is_walkable(x, y):
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                return False
            if (x, y) in self.lava_cells:
                return False
            if (x, y) in extra_blocked:
                return False
            cell = self.grid.get(x, y)
            return cell is None or (x, y) == (gx, gy) or cell.can_overlap()

        def heuristic(x, y):
            return abs(x - gx) + abs(y - gy)

        start = (pos[0], pos[1], agent_dir)
        counter = 0
        heap = [(heuristic(pos[0], pos[1]), counter, 0, start, [])]
        visited: dict = {}

        while heap:
            f, _, g, state, actions = heapq.heappop(heap)
            x, y, d = state

            if (x, y) == (gx, gy):
                if output == "actions":
                    return actions
                directions = []
                cur_d = agent_dir
                for a in actions:
                    if a == 0:
                        cur_d = (cur_d - 1) % 4
                    elif a == 1:
                        cur_d = (cur_d + 1) % 4
                    else:
                        directions.append(_DIR_VECS[cur_d])
                return directions

            if state in visited:
                continue
            visited[state] = g

            ng = g + 1

            # Turn left
            nd = (d - 1) % 4
            ns = (x, y, nd)
            if ns not in visited:
                counter += 1
                heapq.heappush(heap, (ng + heuristic(x, y), counter, ng, ns, actions + [0]))

            # Turn right
            nd = (d + 1) % 4
            ns = (x, y, nd)
            if ns not in visited:
                counter += 1
                heapq.heappush(heap, (ng + heuristic(x, y), counter, ng, ns, actions + [1]))

            # Move forward
            dx, dy = _DIR_VECS[d]
            nx, ny = x + dx, y + dy
            if is_walkable(nx, ny):
                ns = (nx, ny, d)
                if ns not in visited:
                    counter += 1
                    heapq.heappush(
                        heap, (ng + heuristic(nx, ny), counter, ng, ns, actions + [2])
                    )

        return None

    def get_plan(self, pos=None, agent_dir=None, output="actions"):
        """A* planner from the current position to the goal.

        First tries a path that stays at least `lava_buffer` Manhattan distance
        from any lava cell.  Falls back to any lava-free path if the buffered
        route is unreachable.

        Args:
            pos:       (x, y) start; defaults to agent_pos.
            agent_dir: starting direction; defaults to self.agent_dir.
            output:    'actions'    → list of action ints (0=left,1=right,2=forward)
                       'directions' → list of (dx, dy) move vectors (turns omitted)

        Returns:
            Sequence in the requested format, or None if the goal is unreachable.
        """
        if pos is None:
            pos = self.agent_pos
        if agent_dir is None:
            agent_dir = self.agent_dir

        if self.lava_buffer > 0 and self.lava_cells:
            buffer_cells = self._lava_buffer_cells(self.lava_buffer)
            result = self._astar(pos, agent_dir, output, extra_blocked=buffer_cells)
            if result is not None:
                return result

        return self._astar(pos, agent_dir, output, extra_blocked=set())
