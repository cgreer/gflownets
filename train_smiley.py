from collections import defaultdict
import math

from smiley import Env as SmileyEnv
from traj_balance import Trainer


def evaluate_trained(trainer):
    samples = trainer.samples
    model = trainer.model

    # Collect face counts
    counts = defaultdict(int)
    for samp in samples[-2000:]:
        counts[tuple(samp.features)] += 1

    # Face distribution
    # - feature 0 is "smile/frown"
    # - feature 1 is "left eyebrow up/down"
    # - feature 2 is "right eyebrow up/down"
    total_smiley = 0
    total_frown = 0
    print()
    print("  face".ljust(25), "fraction".ljust(10), "count")
    for x in sorted(counts):
        c = counts[x]
        if x[0] == 1: # 0th feature is smile/frown
            symbol = "😊"
            total_smiley += c
        else:
            symbol = "🙁"
            total_frown += c
        print(
            ("  " + symbol + str(x)).ljust(25),
            str(round(c / 2000, 3)).ljust(10),
            c,
        )

    print()
    print("  Z:".ljust(25), round(math.exp(model.logZ), 2))
    print("  % Smiley:".ljust(25), (total_smiley / 2000.0) * 100.0)
    # print("Smiley/frown counts:".ljust(25), total_smiley, total_frown)
    print()


if __name__ == "__main__":
    # Choose/configure environment
    env = SmileyEnv(n_features=3)

    # Train Model
    trainer = Trainer(env=env)
    trainer.train(n_episodes=10_000)
    evaluate_trained(trainer)
    trainer.dashboard()
