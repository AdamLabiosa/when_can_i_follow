import heapq
import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
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


# Wrapper for multi room gridworld env
class gridworldEnv(baseFourRoomsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add in new actions
        # 0: left, 1: right, 2: forward, 3: create plan
        self.action_space = gym.spaces.Discrete(4)

        # step() and reset() return only the image — align the declared space accordingly
        self.observation_space = self.observation_space["image"]

        self.plan: list[int] | None = None
        self.plan_path: list[tuple[int, int]] = []
        self.max_steps = 100

    def step(self, action):
        self.time += 1

        if action == 3:
            self.plan = self.get_plan()
            image_obs = self.gen_obs()['image']
            
            reward = 0
            term = False
            trunc = self.time >= self.max_steps
            
            return image_obs, reward, term, trunc, {}
        else:
            obs, reward, terminated, truncated, info = super().step(action)
            image_obs = obs['image']
            return image_obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self.time = 0

        obs, info = super().reset(*args, **kwargs)
        self._goal_pos = next(
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (cell := self.grid.get(x, y)) is not None and cell.type == 'goal'
        )
        self.plan = None
        self.plan_path = []
        
        image_obs = obs['image']
        return image_obs, info

    def render(self):
        return super().render()

    def get_full_render(self, highlight, tile_size):
        img = baseFourRoomsEnv.get_full_render(self, highlight, tile_size)
        if self.plan_path:
            img = self._overlay_path(img, tile_size, self.plan_path)
        return img

    def get_pov_render(self, tile_size):
        img = baseFourRoomsEnv.get_pov_render(self, tile_size)
        if self.plan_path:
            view_cells = [
                (vx, vy)
                for wx, wy in self.plan_path
                for vx, vy in [self.get_view_coords(wx, wy)]
                if 0 <= vx < self.agent_view_size and 0 <= vy < self.agent_view_size
            ]
            if view_cells:
                img = self._overlay_path(img, tile_size, view_cells)
        return img

    def _overlay_path(
        self,
        img: np.ndarray,
        tile_size: int,
        cells: list[tuple[int, int]],
    ) -> np.ndarray:
        """Return a copy of `img` with a semi-transparent blue overlay on each cell.

        `cells` are (col, row) grid indices where col maps to the x-axis and
        row maps to the y-axis of the rendered image (img[row*ts, col*ts])."""
        img = img.copy()
        h, w = img.shape[:2]
        for cx, cy in cells:
            ymin, ymax = cy * tile_size, (cy + 1) * tile_size
            xmin, xmax = cx * tile_size, (cx + 1) * tile_size
            if ymin < 0 or xmin < 0 or ymax > h or xmax > w:
                continue
            patch = img[ymin:ymax, xmin:xmax].astype(np.float32)
            patch = patch * (1 - _PATH_ALPHA) + _PATH_COLOR_RGB * _PATH_ALPHA
            img[ymin:ymax, xmin:xmax] = patch.astype(np.uint8)
        return img

    def gen_obs(self):
        obs = super().gen_obs()

        if self.plan is not None:
            obs['plan'] = self.plan

            if self.plan_path:
                image = obs['image'].copy()
                for wx, wy in self.plan_path:
                    vx, vy = self.get_view_coords(wx, wy)
                    if 0 <= vx < self.agent_view_size and 0 <= vy < self.agent_view_size:
                        if image[vx, vy, 0] in _OVERWRITABLE_TYPES:
                            image[vx, vy] = _PATH_CELL
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

    def get_plan(self, pos=None, agent_dir=None, output='actions'):
        """
        A* planner from pos to goal over the (x, y, direction) state space.

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

        gx, gy = self._goal_pos

        def is_walkable(x, y):
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                return False
            cell = self.grid.get(x, y)
            return cell is None or (x, y) == (gx, gy) or cell.can_overlap()

        def heuristic(x, y):
            return abs(x - gx) + abs(y - gy)

        # state: (x, y, dir); heap: (f, counter, g, state, actions_so_far)
        start = (pos[0], pos[1], agent_dir)
        counter = 0
        heap = [(heuristic(pos[0], pos[1]), counter, 0, start, [])]
        visited = {}

        while heap:
            f, _, g, state, actions = heapq.heappop(heap)
            x, y, d = state

            if (x, y) == (gx, gy):
                if output == 'actions':
                    return actions
                # Replay actions to extract movement directions only
                directions = []
                cur_d = agent_dir
                for a in actions:
                    if a == 0:
                        cur_d = (cur_d - 1) % 4
                    elif a == 1:
                        cur_d = (cur_d + 1) % 4
                    else:  # forward
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


