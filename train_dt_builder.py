from decision_tree_builder import (
    Env as DTBuilderEnv,
    Datasets,
    Evaluation,
)
from traj_balance import Trainer


def evaluate_trained(trainer):
    samples = trainer.samples
    dataset = trainer.env.dataset

    harvest = []
    for sample in samples:
        tree = sample.tree
        rew = Evaluation.evaluate_tree(tree, dataset)
        harvest.append((rew, tree.root.split_threshold))

    harvest.sort(key=lambda x: x[0], reverse=True)
    for rew, sp in harvest[:30]:
        print(rew, sp)

    print("\nBest", harvest[0])


if __name__ == "__main__":
    # Choose/configure environment

    dataset = Datasets.simple(n_features=1, noise=0.10)
    env = DTBuilderEnv(
        dataset=dataset,
        max_depth=1,
    )

    # Train Model
    trainer = Trainer(env=env)
    trainer.train(n_episodes=4000)
    evaluate_trained(trainer)
    trainer.dashboard()
