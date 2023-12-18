from dataclasses import dataclass, field
from typing import (
    Any,
    List,
)
import numpy as np
import matplotlib.pyplot as plt

from fourier_grid import Env as FourierGrid
from traj_balance import Trainer


def to_empirical(exits, H):
    '''
    Given some samples of episode terminal states, create an empirical
    reward distribution.
    '''
    grid = np.zeros((H, H), 'float')
    for x, y in exits:
        grid[x, y] += 1.0
    grid = grid / grid.sum()
    return grid


def target_error_figure(
    trainer,
    fig_info,
    step,
    lb,
):
    target = trainer.env.reward_distribution
    H = trainer.env.H
    exits = fig_info.exits
    n_episodes = len(exits)
    assert step < n_episodes
    info = []
    for t in range(lb, n_episodes + 1, step):
        rolling = exits[t-lb:t]
        empirical = to_empirical(rolling, H=H)
        error = np.abs(empirical - target).mean()
        info.append((t, error))

    print("episodes".ljust(10), "L1 Error")
    for t, error in info:
        print(
            str(t).ljust(10),
            error,
        )

    # Plot dist + final empirical
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes[0]
    im = ax.imshow(empirical, cmap='gray', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_title("Empirical")
    ax = axes[1]
    im = ax.imshow(env.reward_distribution, cmap='gray', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_title("Target")
    plt.show()


@dataclass
class FigInfo:
    exits: List[Any] = field(init=False)

    def __post_init__(self):
        self.exits = []

    def add_sample(self, state):
        assert state.features[-1] == 1
        self.exits.append((
            state.features[0],
            state.features[1],
        ))


if __name__ == "__main__":
    env = FourierGrid(H=4)

    n_episodes = 10000
    batch_size = 16

    fig_info = FigInfo()

    def post_batch(x):
        # No slices for deque...
        size = len(x.samples)
        for n_samp in range(batch_size):
            samp = x.samples[size - n_samp - 1]
            fig_info.add_sample(samp)

    trainer = Trainer(env=env)
    trainer.train(
        n_episodes=n_episodes,
        batch_size=batch_size,
        lr_model=0.00236, # 0.00236 in paper
        lr_Z=0.0695, # 0.00695 in paper
        temp=1.046, # 1.0458 in paper
        eps=0.02, # 0.00543 in paper
        r_temp=1.5, # 1.5 in paper; "beta"
        max_samples=100,
        post_batch=post_batch,
    )
    trainer.dashboard()

    target_error_figure(
        trainer,
        fig_info,
        step=1000,
        lb=5000,
    )
