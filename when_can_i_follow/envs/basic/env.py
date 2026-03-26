from envs.moving_openings.env import movingOpeningEnv


class basicEnv(movingOpeningEnv):
    """Four-rooms env with plan-on-demand and static openings (doors never move).

    Identical to movingOpeningEnv but door_shift_prob is fixed at 0.0, so the
    layout stays constant throughout an episode.  This isolates the path-following
    and plan-requesting skills from the challenge of dealing with shifting doors.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("door_shift_prob", 0.0)
        super().__init__(*args, **kwargs)
