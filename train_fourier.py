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


def target_error_figure(trainer, fig_info, step, lb):
    target = trainer.env.reward_fxn.rewards
    target = target / target.sum() # normalize

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

    # Plot dist + final seen empirical
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes[0]
    im = ax.imshow(empirical, cmap='gray', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_title("Empirical")
    ax = axes[1]
    im = ax.imshow(target, cmap='gray', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_title("Target")
    plt.show()


@dataclass
class FigInfo:
    exits: List[Any] = field(init=False)

    def __post_init__(self):
        self.exits = []

    def add_sample(self, state):
        assert state.final is True
        self.exits.append((state.row, state.col))


if __name__ == "__main__":
    env = FourierGrid(H=16)

    n_episodes = 150000
    batch_size = 64

    fig_info = FigInfo()

    def post_batch(tr):
        # No slices for deque...
        size = len(tr.samples)
        for n_samp in range(batch_size):
            samp = tr.samples[size - n_samp - 1]
            fig_info.add_sample(samp)

    trainer = Trainer(env=env)
    trainer.train(
        n_episodes=n_episodes,
        batch_size=batch_size,
        # lr_model=0.00112,
        # lr_Z=0.0634,
        # temp=1.046, # 1.0458 in paper
        eps=0.006, # 0.00543 in paper
        # eps=0.10, # 0.00543 in paper
        r_temp=1.5, # 1.5 in paper; "beta"; explore more when > 1
        post_batch=post_batch,
    )
    # trainer.dashboard()

    target_error_figure(
        trainer,
        fig_info,
        step=n_episodes // 20,
        lb=(n_episodes // 5),
    )
