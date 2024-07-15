from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

from typing import Optional

class FrozenLakeMod(FrozenLakeEnv):
    """ Copied from gym/envs/toy_text/frozen_lake.py at https://github.com/openai/gym/blob/dcd185843a62953e27c2d54dc8c2d647d604b635/gym/envs/toy_text/frozen_lake.py,
    then modified to give negative reward for falling in a hole, and to have 70% chance of going the intended direction"""

    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
    ):
        
        super().__init__(render_mode, desc, map_name, is_slippery)

        desc = self.desc
        nrow, ncol = self.nrow, self.ncol

        nA = 4
        nS = nrow * ncol

        LEFT = 0
        DOWN = 1
        RIGHT = 2
        UP = 3

        # Clear this and rewrite it
        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            terminated = bytes(newletter) in b"GH"
            reward = 1.0 if newletter == b"G" else (-1.0 if newletter == "H" else 0.0)  # BW: modified
            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                # BW: modified
                                if b == a:
                                    prob = 0.7
                                else:
                                    prob = 0.15

                                li.append(
                                    (prob, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))


