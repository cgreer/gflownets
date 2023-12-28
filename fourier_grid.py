from dataclasses import dataclass, field
from typing import (
    Any,
    List,
)
from types import SimpleNamespace as Record

import matplotlib.pyplot as plt
import numpy as np
import torch

MIN_STEPS = 1

Tensor = torch.Tensor
Matrix2D = np.array


@dataclass
class RewardFxn:
    H: int = 4
    n: int = 1000
    c: float = -0.50
    d: float = 0.50
    rescale: float = 10.0
    normalize: bool = False

    rewards: Matrix2D = field(init=False)

    def __post_init__(self):
        if self.normalize is True:
            assert self.rescale is None
        if self.rescale is not None:
            assert self.normalize is False

        # Compute rewards for all coords
        print("\nComputing reward fxn")
        rewards = np.zeros((self.H, self.H))
        for row in range(self.H):
            for col in range(self.H):
                rewards[row, col] = self.reward(row, col)

        # Transform
        # - Shift rewards so they are positive
        # - Either rescale or normalize
        # - Ensure everything is positive (required for gflownets)
        #   - gflownets require non-negative rewards
        rewards = rewards - np.min(rewards)
        if self.normalize:
            rewards = rewards / rewards.sum()
        elif self.rescale is not None:
            scale = self.rescale / np.max(rewards)
            rewards = rewards * scale
        rewards = np.maximum(rewards, 0.0)

        # Stash
        print("  min val:", np.min(rewards))
        print("  max val:", np.max(rewards))
        self.rewards = rewards

    def reward(self, x1, x2):
        reward = 0
        gx1 = x1 * (self.d - self.c) / self.H + self.c
        gx2 = x2 * (self.d - self.c) / self.H + self.c
        for k in range(1, self.n + 1):
            reward += np.cos(2 * (4*k/1000) * np.pi * gx1) \
                + np.sin(2 * (4*k/1000) * np.pi * gx1) \
                + np.cos(2 * (4*k/1000) * np.pi * gx2) \
                + np.sin(2 * (4*k/1000) * np.pi * gx2)
        return reward

    def plot(self):
        normalized = self.rewards / self.rewards.sum()
        plt.imshow(normalized, cmap='gray', interpolation='nearest')
        plt.colorbar()
        plt.title("Reward Distribution")
        plt.show()


@dataclass
class State:
    row: int
    col: int
    final: bool
    H: int
    steps: int

    def clone(self) -> 'State':
        return State(
            row=self.row,
            col=self.col,
            final=self.final,
            H=self.H,
            steps=self.steps,
        )

    def terminal(self) -> bool:
        return self.final

    def encode(self) -> Tensor:
        '''
        Encode to NN representation
        '''
        coords = torch.tensor([self.row, self.col]).long()
        return torch.nn.functional.one_hot(coords, self.H).flatten().float()

    def f_mask(self) -> Tensor:
        '''
        Action mask for forwards actions
        - allowed = 1, disallowed = 0
        '''
        # Stopped
        # - Nothing to do...
        # XXX: Should never be called?
        if self.terminal():
            mask = [0, 0, 0, 0, 0]  # [right, left, up, down, terminate]

        # In progress
        else:
            mask = [1, 1, 1, 1, 1]  # [right, left, up, down, terminate]

            if self.steps <= MIN_STEPS:
                mask[-1] = 0

            # Right edge
            # - Can't move right
            if self.col >= self.H - 1:
                mask[0] = 0

            # Left edge
            # - Can't move left
            if self.col <= 0:
                mask[1] = 0

            # Top edge
            # - Can't move up
            if self.row <= 0:
                mask[2] = 0

            # Bottom edge
            # - Can't move down
            if self.row >= self.H - 1:
                mask[3] = 0
        return torch.Tensor(mask).bool()

    def b_mask(self) -> Tensor:
        '''
        Action mask for backwards actions
        - allowed = 1, disallowed = 0
        '''
        # Episode stopped
        # - Only possible action is to undo terminate
        if self.terminal():
            mask = [0, 0, 0, 0, 1]  # [right, left, up, down, terminate]

        # Episode in progress
        # - No way came from a stopped state
        # - Some movements might not be possible if on edges.
        else:
            mask = [1, 1, 1, 1, 0]  # [right, left, up, down, terminate]

            # Left edge
            # - Right not possible
            if self.col <= 0:
                mask[0] = 0

            # Right edge
            # - Left not possible
            if self.col >= self.H - 1:
                mask[1] = 0

            # Bottom edge
            # - Up not possible
            if self.row >= self.H - 1:
                mask[2] = 0

            # Top edge
            # - Down not possible
            if self.row <= 0:
                mask[3] = 0

        # XXX: Delete!
        mask = [1, 1, 1, 1, 1]
        return torch.Tensor(mask).bool()


@dataclass
class Episode:
    history: List[Any] # [state, action, state, ...]
    H: int
    reward_fxn: RewardFxn

    def step(self, action: List[int]):
        state = self.history[-1]
        assert isinstance(state, State)

        # Transition from s to s'
        state_p = state.clone()
        state_p.steps += 1
        assert sum(action) == 1, "Mutually exclusive!?"
        if action[0] == 1: # right
            assert state_p.col < self.H - 1
            state_p.col += 1
        elif action[1] == 1: # left
            assert state_p.col > 0
            state_p.col -= 1
        elif action[2] == 1: # up
            assert state_p.row > 0
            state_p.row -= 1
        elif action[3] == 1: # down
            assert state_p.row < self.H - 1
            state_p.row += 1
        elif action[4] == 1: # terminate
            assert state_p.final is False
            state_p.final = True

        # Add next (a,s) pair to history
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
            return self.reward_fxn.rewards[s.row][s.col]
        else:
            return None


@dataclass
class Env:
    H: int
    n: int = 1000
    c: float = -0.50
    d: float = 0.50

    reward_fxn: Matrix2D = field(init=False)

    def __post_init__(self):
        self.reward_fxn = RewardFxn(
            H=self.H,
            n=self.n,
            c=self.c,
            d=self.d,
            rescale=10.0,
            normalize=False,
        )

    def spawn(self) -> Episode:
        initial_state = State(row=0, col=0, final=False, H=self.H, steps=0)
        return Episode(
            history=[initial_state],
            H=self.H,
            reward_fxn=self.reward_fxn,
        )

    def encoded_state_size(self) -> int:
        return 2 * self.H # ohe of coords

    def encoded_action_size(self) -> int:
        return 5 # right, left, up, down, terminate

    def to_action_idx(self, action: List[int]) -> int:
        assert sum(action) == 1
        return action.index(1)

    def to_action(self, index: int) -> List[int]:
        action = [0, 0, 0, 0, 0]
        action[index] = 1
        return action

    def plot_reward_distribution(self, array: Matrix2D):
        self.reward_fxn.plot()


class Tasks:

    def check_env(self):
        env = Env()
        episode = env.spawn()
        print()
        print(episode)

        actions = ( # r/l/u/d/term
            [1, 0, 0, 0, 0], # right
            [0, 0, 0, 1, 0], # down
            [1, 0, 0, 0, 0], # right
            [0, 0, 0, 1, 0], # down
            [0, 0, 0, 0, 1], # stop
        )
        for action in actions:
            print(f"\nStep: {action}")
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

        st_time = time.time()
        N = 10000
        tot_steps = 0
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
                tot_steps += 1
        elapsed = time.time() - st_time
        print("elapsed:", elapsed)
        print("ep/s:", round(N / elapsed, 2))
        print("steps/s:", round(tot_steps / elapsed, 2))

    def check_reward_fxn(self):
        rfxn = RewardFxn(H=4)
        print(rfxn.rewards)
        rfxn.plot()
        pass


if __name__ == "__main__":
    # Tasks().check_env()
    Tasks().check_rand_episodes()
    # Tasks().check_reward_fxn()
