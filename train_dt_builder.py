from dt_builder import (
    Env as DTBuilderEnv,
    MLPCodec,
    # Datasets,
    Models,
    Evaluation,
    HaystackTask,
    Tasks,
)
from traj_balance import Trainer


def evaluate_trained(trainer, test_dataset):
    # Compile some data about best seen
    results = []
    for reward, sample in trainer.best_samples():
        tree = sample.tree
        results.append((
            reward,
            tree.root.feature,
            tree.root.thresh,
            tree.size(),
            tree,
        ))

    print("\nTop N")
    for rew, feat, thresh, size, tree in results[:30]:
        print(rew, feat, thresh, size)

    print("\nBest", results[0][:4])
    test_acc = Evaluation.evaluate_tree(
        results[0][4],
        test_dataset,
    )
    results[0][4].display()
    print("Test Acc:", test_acc)


def evaluate_random(
    n_features,
    max_depth,
    n_episodes,
    dataset,
):
    import matplotlib.pyplot as pp
    from timing import report_every

    # Generate Data
    xs = []
    ys = []
    max_r = 0.0
    for ep_i in range(n_episodes):
        report_every("random tree", 250)

        # Generate a random tree
        tree = Models.random_tree(
            n_features=n_features,
            max_depth=max_depth,
        )

        # Evaluate it
        acc = Evaluation.evaluate_tree(tree, dataset)
        max_r = max(max_r, acc)
        xs.append(ep_i)
        ys.append(max_r)

    # Plot max(R) v. n_episodes
    f, ax = pp.subplots(2, 1, figsize=(14, 9))
    pp.sca(ax[0])
    pp.plot(xs, ys)
    pp.ylabel('max(R)')
    # ax[0].legend()

    pp.xlabel("Episodes")
    pp.show()


if __name__ == "__main__":
    # Choose/configure environment

    N = 2000
    n_features = 20
    max_depth = 2
    n_episodes = 1000

    haystack = HaystackTask()
    task = haystack.generate(
        N=N,
        n_features=n_features,
        max_depth=max_depth,
    )
    Tasks().inspect_haystack(task)

    # dataset = Datasets.simple(n_features=25, noise=0.10)
    dataset = task.train

    # evaluate_random(n_features, max_depth, n_episodes, dataset)

    env = DTBuilderEnv(
        dataset=dataset,
        max_depth=max_depth,
    )
    codec = MLPCodec(env=env)

    # Train Model
    trainer = Trainer(env=env, codec=codec)
    trainer.train(
        n_episodes=n_episodes,
        batch_size=16,
        temp=1.05,
        eps=0.02,
        # grad_clip=1000,
        # r_temp=0.60,
        r_temp=1 / 512,
        mlp_hidden=256,
        mlp_layers=2,
    )
    evaluate_trained(trainer, task.test)
    trainer.dashboard()
