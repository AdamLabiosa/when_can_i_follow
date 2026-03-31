import heapq
from collections import deque

import gymnasium as gym
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.world_object import Wall
from minigrid.envs.fourrooms import FourRoomsEnv as baseFourRoomsEnv

# Direction vectors indexed by MiniGrid dir encoding: right, down, left, up
_DIR_VECS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# Encoding used to mark plan path cells in the observation image
_PATH_CELL = np.array([OBJECT_TO_IDX["floor"], COLOR_TO_IDX["blue"], 0], dtype=np.uint8)

# Object types that are safe to overwrite with the path marker
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


class movingOpeningEnv(baseFourRoomsEnv):
    """Four-rooms env with plan-on-demand, without any adversary.

    Actions:
        0 — turn left
        1 — turn right
        2 — move forward
        3 — request plan (A* route to goal, delivered after ``plan_delay`` steps)

    Observation keys:
        image          — 7×7×12 stacked egocentric view (current + 3 prior frames, channel-last)
        plan_computing — 1.0 while a plan is being computed (delay not yet elapsed)
        plan_ready     — 1.0 once a completed plan has been delivered
    """

    def __init__(
        self,
        plan_delay: int = 3,
        door_shift_prob: float = 0.0,
        frame_stack: int = 1,
        path_dir_colors: bool = True,
        max_steps: int = 200,
        *args,
        **kwargs,
    ):
        self.door_shift_prob = door_shift_prob
        self._frame_stack = frame_stack
        self.path_dir_colors = path_dir_colors

        # Plan state — initialised properly in reset()
        self._goal_pos: tuple[int, int] = (0, 0)
        self.plan: list[int] | None = None
        self.plan_path: list[tuple[int, int]] = []
        self.plan_request_pos: tuple[int, int] | None = None
        self.plan_request_dir: int | None = None
        self.plan_started_at: int | None = None
        self.plan_delay = plan_delay
        self.time = 0

        super().__init__(*args, **kwargs)

        # Actions: left / right / forward / request-plan
        self.action_space = gym.spaces.Discrete(4)

        # Build frame buffer now that we know the single-frame image shape
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
            "plan_computing": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "plan_ready": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        self.max_steps = max_steps

    def _make_obs(self, image: np.ndarray) -> dict:
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

    def step(self, action):
        self.time += 1

        # Deliver delayed plan if the wait is over
        if self.plan_started_at is not None and self.time - self.plan_started_at >= self.plan_delay:
            if self.plan is not None:
                self.plan_path = self._actions_to_world_coords(
                    self.plan_request_pos, self.plan_request_dir, self.plan
                )
            else:
                self.plan_path = []
            self.plan_started_at = None

        # Potentially shift door positions this step
        self._shift_doors()

        if action == 3:
            # Request a new plan
            self.plan_started_at = self.time
            self.plan_request_pos = tuple(self.agent_pos)
            self.plan_request_dir = self.agent_dir
            self.plan = self.get_plan()

            image_obs = self.gen_obs()["image"]
            trunc = self.time >= self.max_steps
            return self._make_obs(image_obs), 0.0, False, trunc, {}

        # Normal move
        obs, reward, terminated, truncated, info = super().step(action)
        image = obs["image"]

        truncated = truncated or self.time >= self.max_steps
        return self._make_obs(image), reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self.time = 0
        self.plan = None
        self.plan_path = []
        self.plan_request_pos = None
        self.plan_request_dir = None
        self.plan_started_at = None

        obs, info = super().reset(*args, **kwargs)

        # Locate the goal tile
        self._goal_pos = next(
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (cell := self.grid.get(x, y)) is not None and cell.type == "goal"
        )

        # Fill the entire buffer with the first real frame
        first_frame = obs["image"]
        if self._frame_stack > 1:
            self._frame_buffer = deque(
                [first_frame] * self._frame_stack,
                maxlen=self._frame_stack,
            )

        return self._make_obs(first_frame), info

    def _shift_doors(self) -> None:
        """Per-step door shuffling: each wall segment independently shifts with door_shift_prob."""
        if self.door_shift_prob <= 0.0:
            return

        room_w = self.width // 2
        room_h = self.height // 2

        # (fixed_coord, is_vertical, lo, hi) with hi exclusive
        door_slots = [
            (room_w, True,  1,          room_h),            # vertical, top half
            (room_w, True,  room_h + 1, self.height - 1),   # vertical, bottom half
            (room_h, False, 1,          room_w),            # horizontal, left half
            (room_h, False, room_w + 1, self.width - 1),    # horizontal, right half
        ]

        agent_cell = (int(self.agent_pos[0]), int(self.agent_pos[1]))
        occupied = {agent_cell, self._goal_pos}

        for fixed, is_vert, lo, hi in door_slots:
            if self.np_random.random() >= self.door_shift_prob:
                continue

            current_door: tuple[int, int] | None = None
            wall_cells: list[tuple[int, int]] = []
            for pos in range(lo, hi):
                x, y = (fixed, pos) if is_vert else (pos, fixed)
                cell = self.grid.get(x, y)
                if cell is None and (x, y) not in occupied:
                    current_door = (x, y)
                elif cell is not None and cell.type == "wall":
                    wall_cells.append((x, y))

            if current_door is None or not wall_cells:
                continue

            self.grid.set(*current_door, Wall())
            new_door = wall_cells[int(self.np_random.integers(0, len(wall_cells)))]
            self.grid.set(*new_door, None)

    def render(self):
        return super().render()

    def get_full_render(self, highlight, tile_size):
        img = baseFourRoomsEnv.get_full_render(self, highlight, tile_size)
        if self.plan_path:
            if self.path_dir_colors:
                dirs = self._plan_directions()
                colors: list[np.ndarray] | None = [_DIR_COLORS[d] for d in dirs]
            else:
                colors = None
            img = self._overlay_path(img, tile_size, self.plan_path, colors)
        return img

    def get_pov_render(self, tile_size):
        img = baseFourRoomsEnv.get_pov_render(self, tile_size)
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
        """Return the state-channel direction value for each cell in plan_path."""
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
        """Return a copy of `img` with a semi-transparent overlay on each cell.

        If ``cell_colors`` is provided it must have one RGB array per cell;
        otherwise the default blue is used for every cell.
        """
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

    def gen_obs(self):
        obs = super().gen_obs()

        if self.plan is not None and self.plan_path:
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

    @property
    def goal_pos(self):
        return self._goal_pos

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
            if action == 0:
                d = (d - 1) % 4
            elif action == 1:
                d = (d + 1) % 4
            else:
                dx, dy = _DIR_VECS[d]
                x, y = x + dx, y + dy
                coords.append((x, y))
        return coords

    def get_plan(self, pos=None, agent_dir=None, output="actions"):
        """A* planner from pos to goal over the (x, y, direction) state space.

        Args:
            pos:       (x, y) start position; defaults to self.agent_pos.
            agent_dir: starting direction (0=right,1=down,2=left,3=up).
            output:    'actions'    → list of action ints (0=left,1=right,2=forward)
                       'directions' → list of (dx, dy) move vectors (turns omitted)

        Returns:
            Optimal sequence as specified by ``output``, or None if unreachable.
        """
        if pos is None:
            pos = self.agent_pos
        if agent_dir is None:
            agent_dir = self.agent_dir

        gx, gy = self._goal_pos

        def is_walkable(x, y):
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                return False
            cell = self.grid.get(x, y)
            return cell is None or (x, y) == (gx, gy) or cell.can_overlap()

        def heuristic(x, y):
            return abs(x - gx) + abs(y - gy)

        start = (pos[0], pos[1], agent_dir)
        counter = 0
        heap = [(heuristic(pos[0], pos[1]), counter, 0, start, [])]
        visited = {}

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
                    heapq.heappush(heap, (ng + heuristic(nx, ny), counter, ng, ns, actions + [2]))

        return None  # unreachable
