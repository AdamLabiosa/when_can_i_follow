import heapq
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.world_object import Lava
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


# Wrapper for multi room lava env
class lavaEnv(baseFourRoomsEnv):
    def __init__(
        self,
        lava_prob: float = 0.3,
        lava_spread_rate: float = 0.75,
        frame_stack: int = 1,
        path_dir_colors: bool = True,
        lava_buffer: int = 1,
        *args,
        **kwargs,
    ):
        # Store before super().__init__ since _gen_grid may run during init
        self.lava_prob = lava_prob
        self.lava_spread_rate = lava_spread_rate
        self._frame_stack = frame_stack
        self.path_dir_colors = path_dir_colors
        self.lava_buffer = lava_buffer
        self.lava_cells: set[tuple[int, int]] = set()
        self._doorways: list[tuple[int, int]] = []
        self._active_doorway: tuple[int, int] | None = None
        # Pending doorway to spawn lava on (set each episode, spawns per-step)
        self._pending_lava_doorway: tuple[int, int] | None = None
        # Randomized safe path cells — lava avoids these unless forced
        self._safe_path: set[tuple[int, int]] = set()

        super().__init__(*args, **kwargs)

        # Add in new action
        # 0: left, 1: right, 2: forward, 3: create plan
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
            # 1 while a plan computation is in-flight (requested but delay not elapsed)
            "plan_computing": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            # 1 once a completed plan path has been delivered to the agent
            "plan_ready": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        self.plan: list[int] | None = None
        self.plan_path: list[tuple[int, int]] = []
        self.plan_request_pos: tuple[int, int] | None = None
        self.plan_request_dir: int | None = None
        self.max_steps = 200
        self.time = 0

        self.plan_delay = 3

    def _make_obs(self, image: np.ndarray) -> dict:
        """Build the dict observation from a raw image array."""
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

    def _find_doorways(self) -> list[tuple[int, int]]:
        """Return all gap cells in the two interior walls of the four-rooms grid."""
        room_w = self.width // 2
        room_h = self.height // 2
        doorways: list[tuple[int, int]] = []

        # Vertical interior wall at x = room_w
        for y in range(1, self.height - 1):
            if self.grid.get(room_w, y) is None:
                doorways.append((room_w, y))

        # Horizontal interior wall at y = room_h
        for x in range(1, self.width - 1):
            if self.grid.get(x, room_h) is None:
                doorways.append((x, room_h))

        return doorways

    def _place_lava(self, x: int, y: int) -> None:
        self.grid.set(x, y, Lava())
        self.lava_cells.add((x, y))

    def _generate_safe_path(self) -> set[tuple[int, int]]:
        """Randomized path from agent start to goal, routing around the pending lava doorway.

        Uses A* with large random noise so the path wanders rather than taking the
        optimal route.  Returns a set of grid cells that lava should avoid.
        """
        sx, sy = int(self.agent_pos[0]), int(self.agent_pos[1])
        gx, gy = self._goal_pos
        blocked = {self._pending_lava_doorway} if self._pending_lava_doorway else set()
        noise_scale = float(self.width + self.height)  # noise dominates heuristic → random paths

        def is_walkable(x: int, y: int) -> bool:
            if (x, y) in blocked:
                return False
            if not (0 <= x < self.width and 0 <= y < self.height):
                return False
            cell = self.grid.get(x, y)
            return cell is None or (x, y) == (gx, gy) or cell.can_overlap()

        counter = 0
        heap: list = [(0.0, counter, (sx, sy), [(sx, sy)])]
        visited: set[tuple[int, int]] = set()

        while heap:
            _, _, pos, path = heapq.heappop(heap)
            if pos in visited:
                continue
            visited.add(pos)

            if pos == (gx, gy):
                return set(path)

            x, y = pos
            for dx, dy in _DIR_VECS:
                nx, ny = x + dx, y + dy
                npos = (nx, ny)
                if npos not in visited and is_walkable(nx, ny):
                    noise = float(self.np_random.random()) * noise_scale
                    counter += 1
                    heapq.heappush(heap, (noise, counter, npos, path + [npos]))

        # Fallback: no path found avoiding the doorway — try without the constraint
        if blocked:
            blocked = set()
            counter = 0
            heap = [(0.0, counter, (sx, sy), [(sx, sy)])]
            visited = set()
            while heap:
                _, _, pos, path = heapq.heappop(heap)
                if pos in visited:
                    continue
                visited.add(pos)
                if pos == (gx, gy):
                    return set(path)
                x, y = pos
                for dx, dy in _DIR_VECS:
                    nx, ny = x + dx, y + dy
                    npos = (nx, ny)
                    if npos not in visited and is_walkable(nx, ny):
                        noise = float(self.np_random.random()) * noise_scale
                        counter += 1
                        heapq.heappush(heap, (noise, counter, npos, path + [npos]))

        return set()

    def _spread_lava(self) -> None:
        """Grow the lava blob by one cell, never crossing the safe path.

        Lava stops spreading entirely once all reachable neighbours are either
        walls, already lava, the goal, or on the safe path.
        """
        if not self.lava_cells:
            return
        if self.np_random.random() > self.lava_spread_rate:
            return

        gx, gy = self._goal_pos
        off_path: list[tuple[int, int]] = []

        for lx, ly in self.lava_cells:
            for dx, dy in _DIR_VECS:
                nx, ny = lx + dx, ly + dy
                if (nx, ny) in self.lava_cells:
                    continue
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if (nx, ny) == (gx, gy):
                    continue
                if self.grid.get(nx, ny) is not None:
                    continue
                if (nx, ny) in self._safe_path:
                    continue
                off_path.append((nx, ny))

        candidates = list(set(off_path))
        if not candidates:
            return

        idx = int(self.np_random.integers(0, len(candidates)))
        self._place_lava(*candidates[idx])

    def step(self, action):
        self.time += 1

        # Possibly spawn initial lava this step (per-step probability)
        if self._pending_lava_doorway is not None and not self.lava_cells:
            if self.np_random.random() < self.lava_prob:
                self._active_doorway = self._pending_lava_doorway
                self._pending_lava_doorway = None
                self._place_lava(*self._active_doorway)

        # Spread lava before processing the action
        self._spread_lava()

        # If lava spread onto the agent's cell, end the episode immediately
        if tuple(self.agent_pos) in self.lava_cells:
            image_obs = self.gen_obs()['image']
            return self._make_obs(image_obs), 0, True, False, {"lava_death": True}

        if self.plan_started_at is not None and self.time - self.plan_started_at >= self.plan_delay:
            # Give plan to agent (note it is a slightly old plan due to real-time delay)
            if self.plan is not None:
                self.plan_path = self._actions_to_world_coords(
                    self.plan_request_pos, self.plan_request_dir, self.plan
                )
            else:
                self.plan_path = []
            self.plan_started_at = None

        if action == 3:
            # Create plan
            self.plan_started_at = self.time
            self.plan_request_pos = tuple(self.agent_pos)
            self.plan_request_dir = self.agent_dir
            self.plan = self.get_plan()

            image_obs = self.gen_obs()['image']
            reward = 0
            term = False
            trunc = self.time >= self.max_steps

            return self._make_obs(image_obs), reward, term, trunc, {}
        else:
            obs, reward, terminated, truncated, info = super().step(action)
            reward = 1.0 if terminated and reward > 0 else 0.0
            return self._make_obs(obs['image']), reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self.time = 0
        self.plan_started_at = None
        self.lava_cells = set()
        self._doorways = []
        self._active_doorway = None
        self._pending_lava_doorway = None
        self._safe_path: set[tuple[int, int]] = set()

        obs, info = super().reset(*args, **kwargs)
        self._goal_pos = next(
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (cell := self.grid.get(x, y)) is not None and cell.type == 'goal'
        )
        self.plan = None
        self.plan_path = []
        self.plan_request_pos = None
        self.plan_request_dir = None

        # Discover doorways, pick one, then generate a safe path that routes around it
        self._doorways = self._find_doorways()
        if self._doorways:
            idx = int(self.np_random.integers(0, len(self._doorways)))
            self._pending_lava_doorway = self._doorways[idx]
        self._safe_path = self._generate_safe_path()

        # Fill the entire buffer with the first real frame
        first_frame = obs['image']
        if self._frame_stack > 1:
            self._frame_buffer = deque(
                [first_frame] * self._frame_stack,
                maxlen=self._frame_stack,
            )

        return self._make_obs(first_frame), info

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

        ``cells`` are (col, row) grid indices. If ``cell_colors`` is provided it
        must have one RGB array per cell; otherwise the default blue is used.
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

        if self.plan is not None:
            obs['plan'] = self.plan

            if self.plan_path:
                image = obs['image'].copy()
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
                obs['image'] = image

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
        """Replay a list of actions from a starting state and return every world
        cell entered by a forward move (in order)."""
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
        """Return all grid cells within Manhattan distance n of any lava cell."""
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
        """Core A* over (x, y, dir) state space.

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
                if output == 'actions':
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

        return None

    def get_plan(self, pos=None, agent_dir=None, output='actions'):
        """
        A* planner from pos to goal over the (x, y, direction) state space.

        Prefers paths that stay at least `lava_buffer` Manhattan distance from
        any lava cell.  If no such path exists, falls back to any path that
        merely avoids actual lava cells.

        Args:
            pos:       (x, y) start position.
            agent_dir: starting direction (0=right,1=down,2=left,3=up).
                       Defaults to self.agent_dir.
            output:    'actions'    → list of action ints (0=left,1=right,2=forward)
                       'directions' → list of (dx, dy) move vectors (turns omitted)

        Returns:
            Optimal sequence as specified by `output`, or None if unreachable.
        """
        if pos is None:
            pos = self.agent_pos
        if agent_dir is None:
            agent_dir = self.agent_dir

        # First pass: try to stay lava_buffer cells away from any lava
        if self.lava_buffer > 0 and self.lava_cells:
            buffer_cells = self._lava_buffer_cells(self.lava_buffer)
            result = self._astar(pos, agent_dir, output, extra_blocked=buffer_cells)
            if result is not None:
                return result

        # Fallback: just avoid actual lava cells
        return self._astar(pos, agent_dir, output, extra_blocked=set())
