import math
from typing import (
    Any,
    List,
)
from types import SimpleNamespace as Record

from dt_builder import (
    Env as DTBuilderEnv,
    MLPCodec,
    # Datasets,
    Models,
    Evaluation,
    HaystackTask,
    Tasks as BuilderTasks,
)
from traj_balance import Trainer

enu = enumerate


def evaluate_trained(trainer, test_dataset):
    fig_data = []

    # Train
    train_data = Record()
    train_data.model = "DT-GFN"
    train_data.train = True
    train_data.xs = []
    train_data.ys = []
    batch_size = trainer.batch_info[0].size
    max_reward = -1.0
    for i, batch in enu(trainer.batch_info):
        n_ep = i * batch_size
        max_reward = max(batch.max_reward, max_reward)
        train_data.xs.append(n_ep)
        train_data.ys.append(max_reward)
    fig_data.append(train_data)

    # Test
    _, best_state = trainer.best_samples()[0]
    test_acc = Evaluation.evaluate_tree(
        best_state.tree,
        test_dataset,
    )
    test_data = Record()
    test_data.model = "DT-GFN"
    test_data.train = False
    test_data.xs = train_data.xs
    test_data.ys = [test_acc] * len(train_data.ys)
    fig_data.append(test_data)

    return fig_data


def evaluate_random(
    n_features,
    max_depth,
    n_episodes,
    train,
    test,
):
    from timing import report_every

    # Generate Data
    fig_data = []
    for dataset_type in (True, False):
        fdata = Record()
        fdata.model = "RandTrees"
        fdata.train = dataset_type
        fdata.xs = []
        fdata.ys = []
        fig_data.append(fdata)

    max_train_r = 0.0
    max_test_r = 0.0
    for ep_i in range(n_episodes):
        report_every("random tree", 250)

        # Generate a random tree
        tree = Models.random_tree(
            n_features=n_features,
            max_depth=max_depth,
        )

        # Evaluate it
        acc = Evaluation.evaluate_tree(tree, train)
        max_train_r = max(max_train_r, acc)
        fig_data[0].xs.append(ep_i)
        fig_data[0].ys.append(max_train_r)

        acc = Evaluation.evaluate_tree(tree, test)
        max_test_r = max(max_test_r, acc)
        fig_data[1].xs.append(ep_i)
        fig_data[1].ys.append(max_test_r)
    return fig_data


class PerformanceFigure:

    @classmethod
    def build(Cls, data: List[Any]):
        '''
        FigData
            model: str
            train: bool
            xs: List[int]
            ys: List[float]
        '''
        import matplotlib.pyplot as pp
        f, ax = pp.subplots(2, 1, figsize=(14, 9), squeeze=False)

        pp.sca(ax[0, 0])
        for figdata in data:
            if figdata.train:
                continue
            model = figdata.model
            label = f"{model} (test)"
            pp.plot(figdata.xs, figdata.ys, label=label)
            pp.ylabel('Accuracy')

        pp.sca(ax[1, 0])
        for figdata in data:
            if not figdata.train:
                continue
            model = figdata.model
            label = f"{model} (train)"
            pp.plot(figdata.xs, figdata.ys, label=label)
            pp.ylabel('Accuracy')

        ax[0, 0].legend()
        ax[1, 0].legend()

        pp.xlabel("Episodes")
        pp.show()


class Tasks:

    def check_performance_fig(self):
        N = 10_000

        fig_data = []
        for train in (True, False):
            cart = Record()
            cart.model = "CART"
            cart.train = train
            cart.xs = list(range(N))
            acc = 0.99 if train else 0.95
            cart.ys = [acc] * N
            fig_data.append(cart)

            dtgfn = Record()
            dtgfn.model = "DT-GFN"
            dtgfn.train = train
            dtgfn.xs = list(range(N))
            acc = 0.95 if train else 0.92
            dtgfn.ys = []
            for n_ep in range(N):
                progress = (n_ep / N) * 2.0
                saturation = math.tanh(progress * acc)
                dtgfn.ys.append(saturation * acc)
            fig_data.append(dtgfn)
        PerformanceFigure.build(fig_data)


if __name__ == "__main__":
    # Tasks().check_performance_fig()
    # import sys; sys.exit() # noqa

    # Choose/configure environment
    N = 2000
    n_features = 20
    max_depth = 3
    n_episodes = 3000

    haystack = HaystackTask()
    task = haystack.generate(
        N=N,
        n_features=n_features,
        max_depth=max_depth,
    )
    BuilderTasks().inspect_haystack(task)

    # dataset = Datasets.simple(n_features=25, noise=0.10)
    dataset = task.train

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
    trainer.dashboard()

    # Performance Figure
    figdata = []
    d = evaluate_random(
        n_features,
        max_depth,
        n_episodes,
        task.train,
        task.test,
    )
    figdata.extend(d)

    d = evaluate_trained(trainer, task.test)
    figdata.extend(d)

    PerformanceFigure.build(figdata)
