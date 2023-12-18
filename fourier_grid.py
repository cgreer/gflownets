import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import (
    Any,
    List,
)
from types import SimpleNamespace as Record
import torch

Tensor = torch.Tensor
Matrix2D = np.array


@dataclass
class State:
    features: List[int] # [x, y, stop]
    H: int

    def clone(self) -> 'State':
        return State(
            features=self.features[:],
            H=self.H,
        )

    def terminal(self) -> bool:
        return self.features[-1] == 1

    def encode(self) -> Tensor:
        '''
        Encode to NN representation
          - x , y , terminal
        '''
        return torch.tensor(self.features).float()

    def f_mask(self) -> Tensor:
        '''
        Action mask for forwards actions
        - allowed = 1, disallowed = 0
        '''
        # Stopped
        # - Nothing to do...
        # XXX: Should never be called?
        if self.features[-1] == 1:
            mask = [0, 0, 0, 0, 0]  # [right, left, up, down, terminate]

        # In progress
        else:
            mask = [1, 1, 1, 1, 1]  # [right, left, up, down, terminate]

            # Left edge
            # - Can't move left
            if self.features[0] <= 0:
                mask[1] = 0

            # Right edge
            # - Can't move right
            if self.features[0] >= self.H - 1:
                mask[0] = 0

            # Top edge
            # - Can't move up
            if self.features[1] == 0:
                mask[2] = 0

            # Bottom edge
            # - Can't move down
            if self.features[1] == self.H - 1:
                mask[3] = 0
        return torch.Tensor(mask).bool()

    def b_mask(self) -> Tensor:
        '''
        Action mask for backwards actions
        - allowed = 1, disallowed = 0
        '''
        # Episode stopped
        # - Only possible action is to undo terminate
        if self.features[-1] == 1:
            mask = [0, 0, 0, 0, 1]  # [right, left, up, down, terminate]

        # Episode in progress
        # - No way came from a stopped state
        # - Some movements might not be possible if on edges.
        else:
            mask = [1, 1, 1, 1, 0]  # [right, left, up, down, terminate]

            # Left edge
            # - Right not possible
            if self.features[0] <= 0:
                mask[0] = 0

            # Right edge
            # - Left not possible
            if self.features[0] >= self.H - 1:
                mask[1] = 0

            # Top edge
            # - Down not possible
            if self.features[1] == 0:
                mask[3] = 0

            # Bottom edge
            # - Up not possible
            if self.features[1] == self.H - 1:
                mask[2] = 0
        return torch.Tensor(mask).bool()


@dataclass
class Episode:
    history: List[Any] # [state, action, state, ...]
    H: int
    reward_distribution: Matrix2D

    def step(self, action: List[int]):
        state = self.history[-1]
        assert isinstance(state, State)

        # Add next (a,s) pair to history
        state_p = state.clone()
        assert sum(action) == 1, "Mutually exclusive!?"
        if action[0] == 1: # right
            assert state_p.features[0] < self.H - 1
            state_p.features[0] += 1
        elif action[1] == 1: # left
            assert state_p.features[0] > 0
            state_p.features[0] -= 1
        elif action[2] == 1: # up
            assert state_p.features[1] > 0
            state_p.features[1] -= 1
        elif action[3] == 1: # down
            assert state_p.features[1] < self.H - 1
            state_p.features[1] += 1
        elif action[4] == 1: # terminate
            assert state_p.features[2] == 0
            state_p.features[2] = 1
        self.history.append(action)
        self.history.append(state_p)

    def current(self) -> State:
        return self.history[-1]

    def done(self) -> bool:
        return self.history[-1].terminal()

    def n_steps(self) -> int:
        assert len(self.history) # Should always have at least initial state
        return 1 + ((len(self.history) - 1) // 2)

    def steps(self) -> List[Any]:
        steps = []
        history_size = len(self.history)
        for i in range(0, history_size, 2):
            step = Record()
            step.t = i // 2
            step.state = self.history[i]
            step.action_in = self.history[i-1] if i > 0 else None
            step.action_out = self.history[i+1] if i < (history_size - 1) else None
            steps.append(step)
        return steps

    def reward(self) -> float:
        s = self.current()
        if s.terminal():
            x1 = self.history[-1].features[0]
            x2 = self.history[-1].features[1]
            return self.reward_distribution[x2][x1]
        return None


@dataclass
class Env:
    H: int = 4
    n: int = 1000
    c: float = -0.50
    d: float = 0.50
    reward_distribution: Matrix2D = field(init=False)

    def __post_init__(self):
        print("\nComputing true reward distribution")
        self.reward_distribution = self.true_reward_distribution()
        print("...done")

    def spawn(self) -> Episode:
        initial_state = State(features=[0, 0, 0], H=self.H)
        return Episode(
            history=[initial_state],
            H=self.H,
            reward_distribution=self.reward_distribution,
        )

    def encoded_state_size(self) -> int:
        return 3 # x, y, terminal

    def encoded_action_size(self) -> int:
        return 5 # right, left, up, down, terminate

    def to_action_idx(self, action: List[int]) -> int:
        assert sum(action) == 1
        return action.index(1)

    def to_action(self, index: int) -> List[int]:
        action = [0, 0, 0, 0, 0]
        action[index] = 1
        return action

    def reward_calc(self, state):
        x1 = state.features[0]
        x2 = state.features[1]
        reward = 0
        for k in range(1, self.n + 1):
            reward += np.cos(2 * (4*k/1000) * np.pi * self.g(x1)) \
                + np.sin(2 * (4*k/1000) * np.pi * self.g(x1)) \
                + np.cos(2 * (4*k/1000) * np.pi * self.g(x2)) \
                + np.sin(2 * (4*k/1000) * np.pi * self.g(x2))
        return reward

    def g(self, x):
        out = x * (self.d - self.c) / self.H + self.c
        return out

    def true_reward_distribution(self) -> np.ndarray:
        """
        Returns a 2D array representing the true reward distribution
        of the grid. Each element in the array corresponds to the
        reward for that cell.
        """
        reward_distribution = np.zeros((self.H, self.H))
        for x in range(self.H):
            for y in range(self.H):
                state = State(features=[x, y, 0], H=self.H)
                reward_distribution[x, y] = self.reward_calc(state)
        zero_shifted = reward_distribution - np.min(reward_distribution)
        zero_shifted = np.maximum(zero_shifted, 0.0)
        normalized = zero_shifted / zero_shifted.sum()
        return normalized

    def plot_reward_distribution(self, array: Matrix2D):
        plt.imshow(array, cmap='gray', interpolation='nearest')
        plt.colorbar()
        plt.title("Reward Distribution")
        plt.show()


class Tasks:

    def check_env(self):
        env = Env()
        episode = env.spawn()
        print()
        print(episode)

        actions = (
            [1, 0, 0, 0, 0], # right
            [0, 0, 0, 1, 0], # down
            [0, 0, 0, 0, 1], # stop
        )
        for action in actions:
            print("#############################")
            episode.step(action)
            print("state: ", episode.current())
            print("state encoded: ", episode.current().encode())
            print("forward mask: ", episode.current().f_mask())
            print("backward mask: ", episode.current().b_mask())
            print("terminal: ", episode.done())
            print("reward: ", episode.reward())

        print("\nSteps:")
        print("n steps:", episode.n_steps())
        for step in episode.steps():
            print(step)
        print()

    def check_rand_episodes(self):
        from random import choice
        import time
        env = Env(H=64)
        env.plot_reward_distribution(env.reward_distribution)
        st_time = time.time()
        N = 10000
        for _ in range(N):
            ep = env.spawn()
            while not ep.done():
                v_actions = []
                for i, el in enumerate(ep.current().f_mask()):
                    if el >= 0.50:
                        action = [0, 0, 0, 0, 0]
                        action[i] = 1
                        v_actions.append(action)
                action = choice(v_actions)
                ep.step(action)
        elapsed = time.time() - st_time
        print("elapsed:", elapsed)
        print("rate:", round(N / elapsed, 2))


if __name__ == "__main__":
    # Tasks().check_env()
    Tasks().check_rand_episodes()

    # env = Env()
    # episode = env.spawn()
    # print(episode.reward_distribution)
    # env.plot_reward_distribution(env.reward_distribution)
