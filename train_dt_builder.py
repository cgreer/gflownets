from dataclasses import dataclass
from itertools import product
from collections import defaultdict
from typing import (
    Any,
    List,
    ClassVar,
)
from types import SimpleNamespace as Record

import numpy as np
import matplotlib.pyplot as pp
from sklearn.metrics import accuracy_score
from rich import print as rprint

from dt_builder import (
    Env as DTBuilderEnv,
    MLPCodec,
    Models,
    Evaluation,
    HaystackTask,
)
from traj_balance import Trainer
from timing import report_every

enu = enumerate


@dataclass
class EvalTrial:
    id: int
    run_id: str
    task: str
    task_info: Any # partitions, features, depth
    method: str # Name of method
    method_info: Any # Any extra info about the method
    accuracy: float
    f1: float = None
    roc: float = None


class Datasets:

    def iris_binary(
        self,
        scale,
        random_state=None,
        test_size=0.20,
    ):
        '''Iris w/ Setosa Removed'''
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler
        if scale is not None:
            assert len(scale) == 2

        # Load the Iris dataset
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Transform the features using MinMaxScaler
        scaler = MinMaxScaler(feature_range=(scale[0], scale[1]))
        X_scaled = scaler.fit_transform(X)

        X_filtered = []
        y_filtered = []
        for i in range(len(X)):
            if y[i] == 0:
                continue
            X_filtered.append(X_scaled[i])
            if y[i] == 1:
                yp = 0
            else:
                yp = 1
            y_filtered.append(yp)
        X_filtered = np.array(X_filtered, 'float')
        y_filtered = np.array(y_filtered, 'int')

        # Split the dataset into a training set and a test set
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered,
            y_filtered,
            test_size=test_size,
            random_state=random_state,
        )
        return (
            (X_train, y_train),
            (X_test, y_test),
        )


@dataclass
class HaystackEvalFigure:
    data: Any # Dict[key, data]
    figure: Any = None

    @classmethod
    def from_run_data(Cls, trials):
        # Aggregate trials into metric groups
        samples = defaultdict(list)
        for trial in trials:
            key = (
                trial.method,
                trial.task_info.partitions,
                trial.task_info.features,
                trial.task_info.depth,
            )
            samples[key].append(trial)

        # Agg samples into data table
        # - Error
        data = {}
        for key in samples:
            group = Record()
            group.method = key[0]
            group.partitions = key[1]
            group.features = key[2]
            group.depth = key[3]

            samps = samples[key]
            accs = [trial.accuracy for trial in samps]
            errors = [1.0 - x for x in accs]
            group.N = len(samps)
            group.error_mean = np.mean(errors)
            group.error_std = np.std(errors)

            data[key] = group

        fig = Cls(data)
        fig.build()
        return fig

    def enumerate_dimensions(self):
        # Go through all the trial data and pull out settings it was
        # ran over.
        partitions = []
        features = []
        depths = []
        methods = []
        for datum in self.data.values():
            partitions.append(datum.partitions)
            features.append(datum.features)
            depths.append(datum.depth)
            methods.append(datum.method)
        partitions = sorted(list(set(partitions)))
        features = sorted(list(set(features)))
        depths = sorted(list(set(depths)))
        methods = sorted(list(set(methods)))
        return (
            partitions,
            features,
            depths,
            methods,
        )

    def build(self):
        color = None # "tab:green"
        capsize = 5
        partitions, features, depths, methods = self.enumerate_dimensions()

        # Figure represents a single max depth
        assert len(depths) == 1
        max_depth = depths[0]

        # Create subplots
        # - One row/panel for each feature setting
        # - Each panel shows a group of bar charts for each n_part setting.
        n_subplots = len(features)
        fig, axs = pp.subplots(n_subplots, 1, figsize=(10, 11), squeeze=False)
        # fig.suptitle('Model Evaluation on Haystack', fontsize=16, fontweight='bold')
        for sp_row, feat in enu(features):
            groups = [f"|P|={p}" for p in partitions]
            n_groups = len(groups)
            n_methods = len(methods)

            # collect bar chart plotting data
            method_means = []
            method_errors = []
            for method in methods:
                means = []
                errors = []
                for part in partitions:
                    key = (method, part, feat, max_depth)
                    data = self.data[key]
                    means.append(data.error_mean)
                    errors.append(data.error_std)
                method_means.append(means)
                method_errors.append(errors)

            # Setting the positions and width for the bars
            index = np.arange(n_groups)
            bar_width = 0.60 / n_methods
            ax = axs[sp_row, 0]
            for i, method in enu(methods):
                bar_values = method_means[i]
                bar_errors = method_errors[i]
                ax.bar(
                    index + i*bar_width,
                    bar_values,
                    bar_width,
                    yerr=bar_errors,
                    label=method,
                    color=color,
                    capsize=capsize
                )
            ax.set_ylabel('Error')
            ax.set_title(f"Features={feat}")
            # ax.set_xticks(index + bar_width)
            glabel_off = (bar_width * (n_methods - 1)) / 2.0
            print(index)
            print(bar_width)
            print(glabel_off)
            ax.set_xticks(index + glabel_off)
            ax.set_xticklabels(groups)
            ax.set_ylim(0.0, 0.50)

            # Doubling the number of y-axis ticks
            curr_ticks = ax.get_yticks()
            new_step = np.diff(curr_ticks)[0] / 2
            new_ticks = np.arange(curr_ticks[0], curr_ticks[-1] + new_step, new_step)
            ax.set_yticks(new_ticks)

            ax.legend()
        pp.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0)
        self.figure = (fig, axs)

    def show(self):
        pp.show()


@dataclass
class RandTrees:
    model: Any
    training_logs: List[Any]
    n_episodes: int
    n_features: int
    max_depth: int

    NAME: ClassVar[str] = "RandTrees"

    @classmethod
    def train(
        Cls,
        dataset,
        n_episodes,
        n_features,
        max_depth,
    ):
        training_logs = []
        best_acc = 0.0
        best_tree = None
        for ep_i in range(n_episodes):
            report_every("random tree", 250)

            # Generate a random tree
            tree = Models.random_tree(
                n_features=n_features,
                max_depth=max_depth,
            )

            # Annotate nodes w/ outcome data
            # - Necessary to get estimates.
            tree.reannotate(
                observations=dataset[0],
                labels=dataset[1],
            )

            # Evaluate it
            acc = Evaluation.accuracy(tree, dataset)
            if acc > best_acc:
                best_tree = tree
                best_acc = acc

            # Training log
            datapoint = Record()
            datapoint.dataset = "train"
            datapoint.episode = ep_i
            datapoint.accuracy = best_acc
            training_logs.append(datapoint)
        return Cls(
            model=best_tree,
            training_logs=training_logs,
            n_episodes=n_episodes,
            n_features=n_features,
            max_depth=max_depth,
        )

    def evaluate(self, dataset):
        acc = Evaluation.accuracy(self.model, dataset)
        return acc

    def method_info(self):
        return Record(
            n_episodes=self.n_episodes,
            n_features=self.n_features,
            max_depth=self.max_depth,
        )


@dataclass
class CART:
    model: Any
    training_logs: List[Any]

    NAME: ClassVar = "CART"

    @classmethod
    def train(Cls, dataset):
        from sklearn import tree

        # Get datasets
        X, y = Cls.convert_dataset(dataset)

        # Train CART
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, y)
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)

        training_logs = []
        datapoint = Record(dataset="train", episode=None, accuracy=acc)
        training_logs.append(datapoint)

        return Cls(
            model=clf,
            training_logs=training_logs,
        )

    @classmethod
    def convert_dataset(Cl, ds):
        assert isinstance(ds[0], np.ndarray)
        X = ds[0]

        # Convert to "-1" or "1" for DecisionTreeClassifier
        ymod = []
        for val in ds[1]:
            if val >= 0.5:
                ymod.append(1)
            else:
                ymod.append(-1)
        y = np.array(ymod, "int")

        return X, y

    def evaluate(self, dataset):
        X, y = self.convert_dataset(dataset)
        preds = self.model.predict(X)
        acc = accuracy_score(y, preds)
        return acc

    def method_info(self):
        return None


@dataclass
class HumanTree:
    model: Any
    training_logs: List[Any]

    NAME: ClassVar[str] = "HumanTree"

    @classmethod
    def train(
        Cls,
        dataset,
        max_depth,
        n_episodes=1_000_000,
    ):
        env = DTBuilderEnv(
            dataset=dataset,
            max_depth=max_depth,
        )
        n_features = env.n_features()
        training_logs = []
        best_acc = 0.0
        best_tree = None
        user_quit = False
        for ep_i in range(n_episodes):
            if user_quit:
                break

            episode = env.spawn()
            while not episode.done():
                rprint(f"\n\n[green]EPISODE {ep_i} (step {episode.n_steps()})[/green]")
                v_actions = episode.current().valid_f_actions()

                # Display state
                s = episode.current()
                rprint("\n[green]Game State[/green]")
                print("Working split:", s.split)
                print("Stopped:", s.stopped)
                s.tree.game_display(n_features=n_features, max_depth=max_depth)

                # Show user valid actions
                print("\n\nACTIONS:")
                for i, action in enu(v_actions):
                    # print(i, action)
                    print("  ", str(i) + " -", env.pretty_action(action))
                print("  ", "sp <node_id>")
                print("  ", "quit")

                # Get valid choice
                valid_resps = set([str(i) for i in range(len(v_actions))])
                valid_resps.add("quit")
                while True:
                    user_resp = input("\nAction? ").strip()

                    # User wants to scatterplot a node
                    if user_resp.startswith("sp "):
                        _, nid = user_resp.split("sp ", 1)
                        try:
                            s.tree.scatterplot(int(nid))
                        except: # noqa
                            print("scatter failed")
                        continue

                    # Verify valid action
                    if user_resp not in valid_resps:
                        print("Invalid action:", user_resp)
                        continue
                    break

                # User wants to quit?
                # - Quit this episode. Session will quit next loop.
                if user_resp == "quit":
                    user_quit = True
                    break

                # User chose valid action
                act_idx = int(user_resp)
                action = v_actions[act_idx]
                print("Chose", action)
                # action = choice(v_actions)
                episode.step(action)

            # Evaluate final tree
            tree = episode.current().tree
            acc = Evaluation.accuracy(tree, dataset)
            if acc > best_acc:
                best_tree = tree
                best_acc = acc
            print("\nFinal tree:")
            tree.display()
            print("\nacc:", acc)

            # Training log
            datapoint = Record()
            datapoint.dataset = "train"
            datapoint.episode = ep_i
            datapoint.accuracy = best_acc
            training_logs.append(datapoint)

        return Cls(
            model=best_tree,
            training_logs=training_logs,
        )

    def evaluate(self, dataset):
        acc = Evaluation.accuracy(self.model, dataset)
        return acc

    def method_info(self):
        return None


@dataclass
class DTGFN:
    model: Any
    training_logs: List[Any]
    n_episodes: int
    max_depth: int

    NAME: ClassVar[str] = "DT-GFN"

    @classmethod
    def train(
        Cls,
        dataset,
        n_episodes,
        max_depth,
        show_dashboard=True,
    ):

        # Train model
        env = DTBuilderEnv(dataset=dataset, max_depth=max_depth)
        codec = MLPCodec(env=env)
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
        if show_dashboard:
            trainer.dashboard()

        # Create training logs
        training_logs = []
        max_reward = -1.0
        batch_size = trainer.batch_info[0].size
        for i, batch in enu(trainer.batch_info):
            ep_i = i * batch_size
            max_reward = max(batch.max_reward, max_reward)
            datapoint = Record(
                dataset="train",
                episode=ep_i,
                accuracy=max_reward, # XXX: Not nec acc?
            )
            training_logs.append(datapoint)

        # Harvest best tree
        _, best_state = trainer.best_samples()[0]
        best_tree = best_state.tree

        return Cls(
            model=best_tree,
            training_logs=training_logs,
            n_episodes=n_episodes,
            max_depth=max_depth,
        )

    def evaluate(self, dataset):
        return Evaluation.accuracy(self.model, dataset)

    def method_info(self):
        return Record(
            n_episodes=self.n_episodes,
            max_depth=self.max_depth,
        )


class Tasks:

    def check_iris(self):
        train, test = Datasets().iris_binary(scale=[0, 1])

    def create_eval_figure(self):
        run_id = "test_run"
        n_trials = 40
        n_episodes = 3000
        N = 3000
        # partitions = [6, 9, 15]
        partitions = [4, 9, 15]
        # features = [4, 25]
        features = [25]
        depths = [5] # Only 1 allowed; 1^depth must be >= all partitions
        trial_nums = list(range(n_trials))

        # Run trials
        trials = []
        combos = product(partitions, features, depths, trial_nums)
        for n_part, n_feat, depth, trial_num in combos:
            print(f"Running trial: {n_part} {n_feat} {depth} {trial_num}")

            # Generate haystack task
            task = HaystackTask.generate(
                N=N,
                n_partitions=n_part,
                n_features=n_feat,
                max_depth=depth,
            )
            task_info = Record(
                type="haystack",
                partitions=task.n_partitions,
                features=task.n_features,
                depth=task.max_depth,
            )
            print("Generated task")

            # Train/evaluate each method
            # - save a trial for each method
            methods = []

            cart = CART.train(task.train)
            cart_acc = cart.evaluate(task.test)
            methods.append((cart, cart_acc))

            rtree = RandTrees.train(
                task.train,
                n_episodes=n_episodes,
                n_features=n_feat,
                max_depth=depth,
            )
            rtree_acc = rtree.evaluate(task.test)
            methods.append((rtree, rtree_acc))

            '''
            gfn = DTGFN.train(
                task.train,
                n_episodes=n_episodes,
                max_depth=depth,
                show_dashboard=False,
            )
            gfn_acc = gfn.evaluate(task.test)
            methods.append((gfn, gfn_acc))
            '''

            for method, method_acc in methods:
                trial = EvalTrial(
                    id=trial_num,
                    run_id=run_id,
                    task="haystack",
                    task_info=task_info,
                    method=method.NAME,
                    method_info=method.method_info(),
                    accuracy=method_acc,
                )
                trials.append(trial)

        figure = HaystackEvalFigure.from_run_data(trials)
        figure.show()


if __name__ == "__main__":
    # Tasks().check_performance_fig()
    # Tasks().check_iris()
    # Tasks().create_eval_figure()

    '''
    ################
    # Train DT-GFN
    ################
    task = HaystackTask.generate(
        N=3000,
        n_partitions=9,
        n_features=25,
        max_depth=5,
    )

    train, test = task.train, task.test
    # train, test = Datasets().iris_binary(scale=[0, 1])

    dtgfn = DTGFN.train(
        train,
        n_episodes=2000,
        max_depth=5,
        show_dashboard=True,
    )
    dtgfn_acc = dtgfn.evaluate(test)
    print("dt-gfn acc", dtgfn_acc)
    '''

    ################
    # Human Trainer
    ################
    N = 3000
    max_depth = 5
    n_features = 4
    n_partitions = 9

    # Generate hidden tree
    task = HaystackTask.generate(
        N=N,
        n_partitions=n_partitions,
        n_features=n_features,
        max_depth=max_depth,
    )

    # Have human train on task
    human_tree = HumanTree.train(
        task.train,
        max_depth=max_depth,
    )
    human_acc = human_tree.evaluate(task.test)

    # Train CART
    cart = CART.train(task.train)
    cart_acc = cart.evaluate(task.test)

    # Train RandTrees
    rtree = RandTrees.train(
        task.train,
        n_episodes=5000,
        n_features=n_features,
        max_depth=max_depth,
    )
    rtree_acc = rtree.evaluate(task.test)

    rprint("\n[green]True Tree[/green]")
    task.display()

    rprint("\n[green]Scores[/green]")
    print("Truth:".ljust(20), task.test_acc)
    print("Human:".ljust(20), human_acc)
    print("CART:".ljust(20), cart_acc)
    print("RandTrees:".ljust(20), rtree_acc)
    print()
