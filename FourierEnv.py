import numpy as np
import unittest
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import (
    Any,
    List,
    Tuple,
    Optional,
)
from types import SimpleNamespace as Record
import torch

Tensor = torch.Tensor


@dataclass
class State:
    features: List[int] # n binary features, None=unspecified
    H: int

    def clone(self) -> 'State':
        return State(features=self.features[:], H=self.H)

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
        mask = [1, 1, 1, 1, 1]  #[right, left, up, down, terminate]
        if self.features[0] == self.H - 1:
            mask[0] = 0
        if self.features[0] == 0:
            mask[1] = 0
        if self.features[1] == self.H - 1:
            mask[3] = 0
        if self.features[1] == 0:
            mask[2] = 0
        if self.features[-1] == 1:
            mask[-1] = 0
        return torch.Tensor(mask).bool()

    def b_mask(self) -> Tensor:
        '''
        Action mask for backwards actions
        - allowed = 1, disallowed = 0
        '''
        mask = [1, 1, 1, 1, 1]  ##[right, left, up, down, terminate]
        if self.features[0] == self.H - 1:
            mask[1] = 0
        if self.features[0] == 0:
            mask[0] = 0
        if self.features[1] == self.H - 1:
            mask[2] = 0
        if self.features[1] == 0:
            mask[3] = 0
        if self.features[-1] == 0:
            mask[-1] = 0
        return torch.Tensor(mask).bool()


@dataclass
class Episode:
    def __init__(self, history: List[Any], H=4, reward_distribution=None, max_steps=1000):
        self.history = history
        self.H = H
        self.reward_distribution = reward_distribution
        self.max_steps = max_steps

    def step(self, action: List[int]):
        state = self.history[-1]
        assert isinstance(state, State)
        # Add next (a,s) pair to history
        state_p = state.clone() # [right, left, up, down, terminate]
        # [right, left, up, down, terminate]
        if action[0] == 1 and state_p.features[0] < self.H - 1:
            state_p.features[0] += 1
        if action[1] == 1 and state_p.features[0] > 0:
            state_p.features[0] -= 1
        if action[2] == 1 and state_p.features[1] > 0:
            state_p.features[1] -= 1
        if action[3] == 1 and state_p.features[1] < self.H - 1:
            state_p.features[1] += 1
        if action[4] == 1 and state_p.features[2] == 0:
            state_p.features[2] = 1

        self.history.append(action)
        self.history.append(state_p)

    def current(self) -> State:
        return self.history[-1]

    def done(self) -> bool:
        if len(self.history) >= self.max_steps:
            return True
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
        if self.history[-1].features[2] == 1 and self.done():
            x1 = self.history[-1].features[0]
            x2 = self.history[-1].features[1]
            return self.reward_distribution[x2][x1]
        if self.history[-1].features[2] == 0 and self.done():
            return 0
        else:
            return None


@dataclass
class Env:

    def __init__(self, H=4, n=1000, c=-0.5, d=0.5, beta=1.5):
        self.H = H
        self.n = n
        self.c = c
        self.d = d
        self.beta = beta
        self.reward_distribution = self.true_reward_distribution()
        self.state_visits = np.zeros((self.H, self.H))

    def spawn(self) -> Episode:
        initial_state = State(features=[0, 0, 0], H=64)
        return Episode(
            history=[initial_state],
            reward_distribution=self.reward_distribution,
        )

    def encoded_state_size(self) -> int:
        return 3 # x, y, terminal

    def encoded_action_size(self) -> int:
        return 5 # right, left, up, down, terminate

    def to_action_idx(self, action: List[int]) -> int:
        return action.index(1)

    def to_action(self, index: int) -> List[int]:
        action = [0, 0, 0, 0, 0]
        action[index] = 1
        return action
    
    def reward_calc(self, x):
        x1 = x.features[0] 
        x2 = x.features[1] 
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
        Returns a 2D array representing the true reward distribution of the grid.
        Each element in the array corresponds to the reward for that cell.
        """
        reward_distribution = np.zeros((self.H, self.H))

        for x in range(self.H):
            for y in range(self.H):
                state = State(features=[x, y, 0], H=self.H)
                reward_distribution[x, y] = self.reward_calc(state)

        min_val = np.min(reward_distribution)
        max_val = np.max(reward_distribution)
        normalized = (reward_distribution - min_val) / (max_val - min_val)

        return np.power(normalized, self.beta)

    def visualize_array(self, array):
        """
        Visualize a numpy array.

        Parameters:
        array (numpy.ndarray): A 64x64 numpy array.

        Returns:
        None
        """
        plt.imshow(array, cmap='gray', interpolation='nearest')
        plt.colorbar()
        plt.title("Array Visualization")
        plt.show()
    
    def count_state_visits(self, state):
        """
        Count the number of times each state is visited.
        """
        self.state_visits[state.features[1]][state.features[0]] += 1


class Tasks:

    def check_env(self):
        env = Env()
        episode = env.spawn()
        print()
        print(episode)

        for action in ([0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0,0,0,1,0], [0,0,0,0,1]):
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


if __name__ == "__main__":
    Tasks().check_env()
    env = Env()
    episode = env.spawn()
    # print(episode.reward_distribution)
    # env.visualize_array(env.reward_distribution)

        

