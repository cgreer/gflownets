from dataclasses import dataclass
from typing import (
    Any,
    List,
    # Dict,
    Tuple,
    Optional,
)
from types import SimpleNamespace as Record

import torch


Action = Tuple[int, int] # feature, val
Tensor = torch.Tensor


@dataclass
class State:
    features: List[Optional[int]] # n binary features, None=unspecified

    def clone(self) -> 'State':
        return State(features=self.features[:])

    def reward(self) -> float:
        if not self.terminal():
            return None

        # 0th feature is smile/frown
        if self.features[0] == 1:
            return 2.0
        else:
            return 1.0

    def terminal(self) -> bool:
        return None not in self.features

    def encode(self) -> Tensor:
        '''
        Encode to NN representation
          - One-hot encode each binary feature, plus an additional bit
            for if the feature has been specified yet.
        '''
        # XXX: Redo efficiently
        enc = []
        for val in self.features:
            if val is None:
                enc.append(0) # ohe
                enc.append(0) # ohe
                enc.append(0) # specified indicator
            elif val == 0:
                enc.append(1)
                enc.append(0)
                enc.append(1)
            elif val == 1:
                enc.append(0)
                enc.append(1)
                enc.append(1)
            else:
                raise KeyError()
        return torch.tensor(enc).float()

    def f_mask(self) -> Tensor:
        '''
        Action mask for forwards actions
        - allowed = 1, disallowed = 0
        '''
        mask = []
        for val in self.features:
            if val is None:
                mask.extend([1, 1])
            else:
                mask.extend([0, 0])
        return torch.Tensor(mask).bool()

    def b_mask(self) -> Tensor:
        '''
        Action mask for backwards actions
        - allowed = 1, disallowed = 0
        '''
        mask = []
        for val in self.features:
            if val is None:
                mask.extend([0, 0])
            elif val == 0:
                mask.extend([1, 0])
            elif val == 1:
                mask.extend([0, 1])
            else:
                raise KeyError()
        return torch.Tensor(mask).bool()


@dataclass
class Episode:
    n_features: int
    history: List[Any] # Action | State

    def step(self, action: Action):
        state = self.history[-1]
        assert isinstance(state, State)

        # Sanity check invalid action
        feature, val = action
        assert state.features[feature] is None
        assert val in (0, 1)

        # Add next (a,s) pair to history
        state_p = state.clone()
        state_p.features[feature] = val
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
        if not self.done():
            return None
        return self.history[-1].reward()


@dataclass
class Env:
    n_features: int

    def spawn(self) -> Episode:
        initial_state = State(features=[None] * self.n_features)
        return Episode(
            n_features=self.n_features,
            history=[initial_state],
        )

    def encoded_state_size(self) -> int:
        return self.n_features * 3 # ohe + indicator each

    def encoded_action_size(self) -> int:
        return self.n_features * 2 # ohe each

    def to_action_idx(self, action: Action) -> int:
        feature, val = action
        return (2*feature) + val

    def to_action(self, index: int) -> Action:
        feature, val = divmod(index, 2) # 2 cuz binary features
        return (feature, val)


class Tasks:

    def check_env(self):
        env = Env(n_features=3)
        episode = env.spawn()
        print()
        print(episode)
        for action in ((0, 1), (1, 0), (2, 1)):
            episode.step(action)
            print(
                episode.current(),
                episode.current().encode(),
                episode.current().f_mask(),
                episode.current().b_mask(),
                episode.done(),
                episode.current().reward(),
            )

        print("\nSteps:")
        print("n steps:", episode.n_steps())
        for step in episode.steps():
            print(step)
        print()


if __name__ == "__main__":
    Tasks().check_env()
