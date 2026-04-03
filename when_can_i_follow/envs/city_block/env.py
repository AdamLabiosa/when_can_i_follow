import heapq
from collections import deque

import gymnasium as gym
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv

# Direction vectors indexed by MiniGrid dir encoding: right, down, left, up
_DIR_VECS = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# Object types that are safe to overwrite with the path marker in the obs image
_OVERWRITABLE_TYPES = {OBJECT_TO_IDX["empty"], OBJECT_TO_IDX["floor"]}

# RGB color and alpha used to paint the path overlay onto rendered frames
_PATH_COLOR_RGB = np.array([30, 144, 255], dtype=np.float32)  # dodger blue
_PATH_ALPHA = 0.45

# State-channel values that encode movement direction in path cells
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


class cityBlockEnv(MiniGridEnv):
    """NxN grid laid out as city blocks separated by 1-cell-wide streets.

    The world is divided into solid city blocks (walls) and streets (walkable
    1-cell-wide corridors). Lava is scattered on street segments between
    intersections. The agent must navigate from a random intersection to a
    goal intersection.

    The key mechanic: the planner (A*) only knows about lava cells the agent
    has already observed. Unseen lava is treated as passable, so the planner
    may route the agent through hidden danger zones. At least one lava-free
    path from start to goal is always guaranteed by construction.

    Actions: 0=turn left, 1=turn right, 2=move forward, 3=request plan.

    Parameters
    ----------
    grid_size    : Full grid side length including outer walls.
    block_size   : Side length of each city block in cells.  Streets are
                   always 1 cell wide and fall every (block_size + 1) cells.
    lava_density : Probability that each non-protected street-segment cell
                   gets a lava tile.
    plan_delay   : Steps between a plan request and its delivery.
    frame_stack  : Number of consecutive frames to stack (1 = disabled).
    path_dir_colors : Encode movement direction in path overlay colors.
    min_goal_dist   : Minimum intersection-graph distance between start and
                      goal (in intersection hops).
    """

    def __init__(
        self,
        grid_size: int = 19,
        block_size: int = 3,
        lava_density: float = 0.25,
        plan_delay: int = 3,
        frame_stack: int = 1,
        path_dir_colors: bool = True,
        min_goal_dist: int = 3,
        max_steps: int = 500,
        lava_movement_prob: float = 0.0,
        **kwargs,
    ):
        self.grid_size = grid_size
        self.block_size = block_size
        self.lava_density = lava_density
        self.plan_delay = plan_delay
        self._frame_stack = frame_stack
        self.path_dir_colors = path_dir_colors
        self.min_goal_dist = min_goal_dist
        self.lava_movement_prob = lava_movement_prob

        # Runtime state — set before super().__init__ so _gen_grid runs safely
        self.lava_cells: set[tuple[int, int]] = set()
        self.seen_lava: set[tuple[int, int]] = set()
        self._goal_pos: tuple[int, int] = (grid_size // 2, grid_size // 2)
        self.plan_started_at: int | None = None
        self.time = 0
        self._street_cols: list[int] = []
        self._street_rows: list[int] = []

        mission_space = MissionSpace(mission_func=lambda: "reach the green goal")
        super().__init__(
            mission_space=mission_space,
            width=grid_size,
            height=grid_size,
            max_steps=max_steps,
            **kwargs,
        )

        self.action_space = gym.spaces.Discrete(4)

        single_img_shape = self.observation_space["image"].shape
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
            "plan_ready":     gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        self.plan: list[int] | None = None
        self.plan_path: list[tuple[int, int]] = []
        self.plan_request_pos: tuple[int, int] | None = None
        self.plan_request_dir: int | None = None

    # ------------------------------------------------------------------
    # Grid layout helpers
    # ------------------------------------------------------------------

    def _compute_street_positions(self) -> None:
        """Populate _street_cols and _street_rows for the current grid_size."""
        period = self.block_size + 1
        self._street_cols = list(range(1, self.grid_size - 1, period))
        self._street_rows = list(range(1, self.grid_size - 1, period))

    def _is_street_col(self, x: int) -> bool:
        return x in self._street_col_set

    def _is_street_row(self, y: int) -> bool:
        return y in self._street_row_set

    def _is_street_cell(self, x: int, y: int) -> bool:
        return self._is_street_col(x) or self._is_street_row(y)

    def _is_intersection(self, x: int, y: int) -> bool:
        return self._is_street_col(x) and self._is_street_row(y)

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self._compute_street_positions()
        self._street_col_set = set(self._street_cols)
        self._street_row_set = set(self._street_rows)

        # Place walls for every city block interior cell
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                if not self._is_street_cell(x, y):
                    self.grid.set(x, y, Wall())

        # All intersections are always clear (street × street)
        intersections = [(x, y) for x in self._street_cols for y in self._street_rows]
        n = len(intersections)

        # Pick start and goal separated by at least min_goal_dist hops
        nc = len(self._street_cols)
        nr = len(self._street_rows)
        start_ci = int(self.np_random.integers(0, nc))
        start_ri = int(self.np_random.integers(0, nr))
        for _ in range(200):
            goal_ci = int(self.np_random.integers(0, nc))
            goal_ri = int(self.np_random.integers(0, nr))
            dist = abs(goal_ci - start_ci) + abs(goal_ri - start_ri)
            if dist >= max(self.min_goal_dist, 1):
                break

        sx, sy = self._street_cols[start_ci], self._street_rows[start_ri]
        gx, gy = self._street_cols[goal_ci], self._street_rows[goal_ri]

        self.agent_pos = np.array([sx, sy])
        self.agent_dir = 0  # facing right

        self.grid.set(gx, gy, Goal())
        self._goal_pos = (gx, gy)
        self.mission = "reach the green goal"

        # Scatter lava, guaranteeing at least one safe route
        self._generate_lava((sx, sy), (gx, gy))

    # ------------------------------------------------------------------
    # Lava generation
    # ------------------------------------------------------------------

    def _random_intersection_path(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Random DFS over intersection graph; returns a non-optimal winding path.

        At each intersection the four cardinal neighbours are shuffled before
        exploration, so the safe route is unpredictable and typically much
        longer than the Manhattan-optimal route.  Backtracking guarantees the
        goal is always reached on the fully-connected grid.
        """
        sc = self._street_cols
        sr = self._street_rows
        col_idx = {c: i for i, c in enumerate(sc)}
        row_idx = {r: i for i, r in enumerate(sr)}

        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Each stack frame: (current node, iterator over unvisited neighbours)
        visited = {start}
        path = [start]

        def neighbours(cx: int, cy: int):
            ci, ri = col_idx[cx], row_idx[cy]
            order = self.np_random.permutation(len(deltas))
            for idx in order:
                di, dj = deltas[idx]
                ni, nj = ci + di, ri + dj
                if 0 <= ni < len(sc) and 0 <= nj < len(sr):
                    nb = (sc[ni], sr[nj])
                    if nb not in visited:
                        yield nb

        stack: list = [neighbours(start[0], start[1])]

        while stack:
            try:
                nb = next(stack[-1])
                visited.add(nb)
                path.append(nb)
                if nb == goal:
                    return path
                stack.append(neighbours(nb[0], nb[1]))
            except StopIteration:
                stack.pop()
                if path:
                    path.pop()

        return [start]  # unreachable on a connected grid

    def _segment_cells(
        self, a: tuple[int, int], b: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """All grid cells (inclusive) on the straight segment between adjacent intersections."""
        ax, ay = a
        bx, by = b
        if ax == bx:
            return [(ax, y) for y in range(min(ay, by), max(ay, by) + 1)]
        return [(x, ay) for x in range(min(ax, bx), max(ax, bx) + 1)]

    def _generate_lava(
        self, start: tuple[int, int], goal: tuple[int, int]
    ) -> None:
        """Scatter lava on street-segment cells, keeping one path guaranteed safe."""
        # Cells along the guaranteed path must stay lava-free
        path = self._random_intersection_path(start, goal)
        protected: set[tuple[int, int]] = set()
        for i in range(len(path) - 1):
            protected.update(self._segment_cells(path[i], path[i + 1]))
        protected.add(start)
        protected.add(goal)

        intersection_set = set(
            (x, y) for x in self._street_cols for y in self._street_rows
        )

        self.lava_cells = set()
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                if not self._is_street_cell(x, y):
                    continue   # city block interior — already a Wall
                if (x, y) in protected:
                    continue
                if (x, y) in intersection_set:
                    continue   # never block intersections with lava
                if self.np_random.random() < self.lava_density:
                    self.grid.set(x, y, Lava())
                    self.lava_cells.add((x, y))

    # ------------------------------------------------------------------
    # Seen-lava tracking
    # ------------------------------------------------------------------

    def _update_seen_lava(self) -> None:
        """Add any lava cells that fall inside the agent's view rectangle."""
        topX, topY, botX, botY = self.get_view_exts()
        for wx in range(topX, botX):
            for wy in range(topY, botY):
                if 0 <= wx < self.width and 0 <= wy < self.height:
                    cell = self.grid.get(wx, wy)
                    if cell is not None and cell.type == "lava":
                        self.seen_lava.add((wx, wy))

    # ------------------------------------------------------------------
    # Lava dynamics
    # ------------------------------------------------------------------

    def _step_lava(self) -> None:
        """Move each lava tile independently with probability epsilon.

        Each lava picks one random cardinal direction; if the target cell is a
        valid (non-intersection) street cell that is not already occupied by
        another lava, the goal, or the agent, the lava slides there.
        Lavas are processed in a shuffled order so no single tile has priority.
        """
        if self.lava_movement_prob <= 0.0:
            return

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        agent_pos = tuple(self.agent_pos)

        lava_list = list(self.lava_cells)
        self.np_random.shuffle(lava_list)

        # Work on a mutable copy so moves within the same step are respected.
        occupied = set(self.lava_cells)

        for (lx, ly) in lava_list:
            if (lx, ly) not in occupied:
                continue  # another lava slid here this step

            if self.np_random.random() >= self.lava_movement_prob:
                continue  # this lava stays put

            di = int(self.np_random.integers(0, 4))
            dx, dy = directions[di]
            nx, ny = lx + dx, ly + dy

            # Must stay inside the inner grid
            if nx < 1 or ny < 1 or nx >= self.width - 1 or ny >= self.height - 1:
                continue
            # Must be a street cell (not a block interior)
            if not self._is_street_cell(nx, ny):
                continue
            # Intersections are lava-free by convention
            if self._is_intersection(nx, ny):
                continue
            # No overlap with another lava
            if (nx, ny) in occupied:
                continue
            # Don't land on the goal or the agent
            if (nx, ny) == self._goal_pos or (nx, ny) == agent_pos:
                continue

            # Commit the move
            occupied.discard((lx, ly))
            occupied.add((nx, ny))
            self.grid.set(lx, ly, None)
            self.grid.set(nx, ny, Lava())

        self.lava_cells = occupied

        # Seen-lava cache: drop entries where lava has since moved away,
        # then reveal any lava that drifted into the agent's current view.
        self.seen_lava = {pos for pos in self.seen_lava if pos in self.lava_cells}
        self._update_seen_lava()

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        self.time = 0
        self.plan_started_at = None
        self.lava_cells = set()
        self.seen_lava = set()
        self.plan = None
        self.plan_path = []
        self.plan_request_pos = None
        self.plan_request_dir = None

        obs, info = super().reset(*args, **kwargs)

        # Reveal lava visible from the starting position
        self._update_seen_lava()

        first_frame = obs["image"]
        if self._frame_stack > 1:
            self._frame_buffer = deque(
                [first_frame] * self._frame_stack,
                maxlen=self._frame_stack,
            )

        return self._make_obs(first_frame), info

    def step(self, action):
        self.time += 1

        # Deliver a pending plan once the delay has elapsed
        if (
            self.plan_started_at is not None
            and self.time - self.plan_started_at >= self.plan_delay
        ):
            if self.plan is not None:
                self.plan_path = self._actions_to_world_coords(
                    self.plan_request_pos, self.plan_request_dir, self.plan
                )
            else:
                self.plan_path = []
            self.plan_started_at = None

        if action == 3:
            self.plan_started_at = self.time
            self.plan_request_pos = tuple(self.agent_pos)
            self.plan_request_dir = self.agent_dir
            self.plan = self.get_plan()
            image_obs = self.gen_obs()["image"]
            reward = -0.01
            return self._make_obs(image_obs), reward, False, self.time >= self.max_steps, {}

        obs, reward, terminated, truncated, info = super().step(action)

        # Expand the seen-lava cache after the agent has moved
        self._update_seen_lava()

        # Stochastically drift lava tiles (no-op when epsilon == 0)
        if not terminated:
            self._step_lava()

        truncated = self.time >= self.max_steps
        reward = 1.0 if terminated and reward > 0 else 0.0
        return self._make_obs(obs["image"]), reward, terminated, truncated, info

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

    # ------------------------------------------------------------------
    # Observation overlay
    # ------------------------------------------------------------------

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
    # Rendering
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
        img = img.copy()
        for i, (cx, cy) in enumerate(cells):
            color = cell_colors[i] if cell_colors is not None else _PATH_COLOR_RGB
            ymin, ymax = cy * tile_size, (cy + 1) * tile_size
            xmin, xmax = cx * tile_size, (cx + 1) * tile_size
            if ymin < 0 or xmin < 0 or ymax > img.shape[0] or xmax > img.shape[1]:
                continue
            patch = img[ymin:ymax, xmin:xmax].astype(np.float32)
            img[ymin:ymax, xmin:xmax] = (
                patch * (1 - _PATH_ALPHA) + color * _PATH_ALPHA
            ).astype(np.uint8)
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

    def _astar(
        self,
        pos: tuple,
        agent_dir: int,
        output: str = "actions",
    ):
        """A* over (x, y, dir) state space using only seen_lava as obstacles.

        Unseen lava is treated as passable (Lava.can_overlap() == True), so
        the planner may route through it.  Only lava cells the agent has
        already observed are avoided.
        """
        gx, gy = self._goal_pos

        def is_walkable(x: int, y: int) -> bool:
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                return False
            if (x, y) in self.seen_lava:
                return False
            cell = self.grid.get(x, y)
            return cell is None or (x, y) == (gx, gy) or cell.can_overlap()

        def heuristic(x: int, y: int) -> int:
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
                cur_d = agent_dir
                directions = []
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
        """A* from current position to goal, avoiding only seen lava.

        The planner has no knowledge of lava cells the agent has not yet
        observed — those cells are treated as passable.  This means the
        plan may route through hidden lava; the agent should re-plan upon
        encountering unexpected obstacles.

        Args:
            pos:       (x, y) start; defaults to agent_pos.
            agent_dir: starting direction; defaults to self.agent_dir.
            output:    'actions'    → list of action ints (0=left,1=right,2=forward)
                       'directions' → list of (dx, dy) move vectors

        Returns:
            Sequence in the requested format, or None if the goal is unreachable
            given the agent's current knowledge.
        """
        if pos is None:
            pos = self.agent_pos
        if agent_dir is None:
            agent_dir = self.agent_dir
        return self._astar(pos, agent_dir, output)
