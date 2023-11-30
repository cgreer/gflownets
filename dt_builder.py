from random import random, choice
from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import (
    Any,
    List,
    # Dict,
    Tuple,
    Optional,
)
from types import SimpleNamespace as Record

import torch

enu = enumerate

Tensor = torch.Tensor
NodeID = int
Action = Tuple[int, float] # feature, thresh
Splitting = Tuple[int, int, float] # NodeIdx, Feature, Threshold

THRESH_SIZE = 21


def bucket_size(min_value, max_value, buckets):
    return (max_value - min_value) / buckets


def quantize(value, min_value=0.0, max_value=1.0, buckets=21):
    # XXX: Double check this
    if value < min_value:
        return 0
    if value >= max_value:
        return buckets - 1
    b_size = bucket_size(min_value, max_value, buckets)
    return int((value - min_value) / b_size)


def discretize(bucket, min_value=0.0, max_value=1.0, buckets=21):
    # XXX: Make midpoint?
    b_size = bucket_size(min_value, max_value, buckets)
    return min_value + (bucket * b_size)


@dataclass
class TreeNode:
    id: int
    depth: int
    parent: Optional[NodeID]
    outcomes: List[int]
    split_feature: Optional[int] = None
    split_threshold: Optional[float] = None
    pass_id: Optional[NodeID] = None
    fail_id: Optional[NodeID] = None

    def copy(self):
        return TreeNode(
            id=self.id,
            depth=self.depth,
            parent=self.parent,
            outcomes=self.outcomes[:],
            split_feature=self.split_feature,
            split_threshold=self.split_threshold,
            pass_id=self.pass_id,
            fail_id=self.fail_id,
        )

    @property
    def split(self):
        return self.split_feature is not None


@dataclass
class Tree:
    max_depth: int # Depth defined as n splits to reach leaf.
    n_outcomes: int
    nodes: List[TreeNode]

    @property
    def root(self):
        return self.nodes[0]

    @classmethod
    def dense(Cls, max_depth: int, n_outcomes=2):
        id_counter = count()

        # Seed bfs expansion
        queue = deque()
        root = TreeNode(
            id=next(id_counter),
            depth=0,
            parent=None,
            outcomes=[0] * n_outcomes,
        )
        queue.append(root)

        # Breadth-first expansion
        nodes = [] # List[TreeNode]
        while queue:
            node = queue.popleft()

            # Do something with node here
            nodes.append(node)

            # Split
            # - Don't split if at max depth
            # - Add pass/fail children
            if node.depth >= max_depth:
                continue
            pchild = TreeNode(
                id=next(id_counter),
                depth=node.depth + 1,
                parent=node.id,
                outcomes=[0] * n_outcomes,
            )
            fchild = TreeNode(
                id=next(id_counter),
                depth=node.depth + 1,
                parent=node.id,
                outcomes=[0] * n_outcomes,
            )
            node.pass_id = pchild.id
            node.fail_id = fchild.id
            queue.append(pchild)
            queue.append(fchild)

        tree = Tree(
            max_depth=max_depth,
            n_outcomes=n_outcomes,
            nodes=nodes,
        )
        return tree

    def copy(self):
        nodes = [x.copy() for x in self.nodes]
        return Tree(
            max_depth=self.max_depth,
            n_outcomes=self.n_outcomes,
            nodes=nodes,
        )

    def split(self, node_id: NodeID, feature: int, thresh: float):
        pnode = self.nodes[node_id]
        assert pnode.pass_id is not None
        assert pnode.fail_id is not None
        pnode.split_feature = feature
        pnode.split_threshold = thresh

    def prune(self):
        raise NotImplementedError()

    def leaf(self, node_id: NodeID):
        '''
        A leaf is a node that:
        - has a parent w/ a split defined (OR is root) AND
        - has no split defined itself.

        Note that, in a dense tree, it's possible to have children but
        no split defined, since all the nodes up to the max depth are
        allocated/created ahead of time before any splitting occurs.
        '''
        pid = self.nodes[node_id].parent
        am_root = pid is None
        split_defined = self.nodes[node_id].split
        if am_root:
            if not split_defined:
                return True
        else:
            if self.nodes[pid].split and not split_defined:
                return True
        return False

    def bfs(self):
        '''
        Returns "active" nodes in tree. Any descendants of leaf nodes
        that weren't part of splits won't be returned.
        '''
        queue = deque()
        visited = set() # XXX: Necessary for tree?

        # Seed w/ root
        queue.append(self.root)
        visited.add(self.root.id)

        # Search
        results = [] # List[TreeNode]
        while queue:
            node = queue.popleft()

            # Do something with node here
            results.append(node)

            # Add neighbors to queue
            # - Added to right side
            if self.leaf(node.id):
                continue
            for nnode_id in (node.pass_id, node.fail_id):
                nnode = self.nodes[nnode_id]
                if nnode.id in visited:
                    continue
                queue.append(nnode)
                visited.add(nnode.id)
        return results

    def traverse(self, x):
        node = self.root
        while not self.leaf(node.id):
            if x[node.split_feature] <= node.split_threshold:
                node = self.nodes[node.pass_id]
            else:
                node = self.nodes[node.fail_id]
        return node

    def enrich_outcomes(self, inputs, labels):
        # XXX: make efficient via using only one pass
        assert len(inputs) == len(labels)
        for i in range(len(inputs)):
            leaf = self.traverse(inputs[i])
            leaf.outcomes[labels[i]] += 1

    def estimate(self, x):
        # XXX: Dirichlet Smoothing?
        # - Hierarchichal updates or just one update?
        #   - Discuss w/ Dustin...
        leaf = self.traverse(x)
        tot_samples = sum(leaf.outcomes)
        est = [x / tot_samples for x in leaf.outcomes]
        return est


@dataclass
class State:
    n_features: int
    tree: Tree

    def clone(self) -> 'State':
        tree = self.tree.copy()
        return State(
            n_features=self.n_features,
            tree=tree,
        )

    def terminal(self) -> bool:
        # XXX: Redo !!
        if self.tree.leaf(0):
            return False
        else:
            return True

    def encode(self) -> Tensor:
        '''
        Encode to NN representation

        For each node in tree:
        - feature ohe (or 0.0 if no split) ; F
        - thresh ohe (or 0.0 if no split) ; F
        - split indicator (1.0 if split else 0.0) ; 1

        The size of each node's encoding is (2F + 1). The size of the
        entire tree's encoding is N x (2F + 1).

        '''
        F = self.n_features
        node_enc_size = (2 * F) + 1
        encoding = []
        for node in self.tree.nodes: # bfs order of ALL nodes
            enc = [0.0] * node_enc_size
            if node.split_feature is not None:
                enc[node.split_feature * 2 * F] = 1.0
                enc[node.split_feature * 2 * F + 1] = node.split_threshold
                enc[-1] = 1.0 # indicator
            encoding.extend(enc)
        return torch.tensor(encoding).float()

    def f_mask(self) -> Tensor:
        '''
        Action mask for forwards actions
        - allowed = 1, disallowed = 0
        '''
        if self.terminal():
            mask = [0] * THRESH_SIZE
        else:
            mask = [1] * THRESH_SIZE
        return torch.Tensor(mask).bool()

    def b_mask(self) -> Tensor:
        '''
        Action mask for backwards actions
        - allowed = 1, disallowed = 0
        '''
        mask = [0] * THRESH_SIZE
        if self.terminal():
            # What is the action that lead to that thresh?
            # XXX: Replace with ANY last splits
            thresh = self.tree.root.split_threshold
            thresh_idx = quantize(thresh, buckets=THRESH_SIZE)
            mask[thresh_idx] = 1
        return torch.Tensor(mask).bool()


class Evaluation:

    @classmethod
    def evaluate_tree(Cls, tree, dataset):
        # accuracy + penalty for train/infer time
        # - accuracy ~ logloss
        inputs, labels = dataset
        tree.enrich_outcomes(inputs, labels)
        n_samples = len(inputs)
        acc = 0
        for i in range(n_samples):
            est = tree.estimate(inputs[i])
            best_idx = 0
            best_val = est[0]
            idx = 0
            for val in est[1:]:
                idx += 1
                if val >= best_val:
                    best_val = val
                    best_idx = idx
            if best_idx == labels[i]:
                acc += 1
        return acc / n_samples


@dataclass
class BaseEpisode:

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


@dataclass
class Episode(BaseEpisode):
    dataset: Any
    history: List[Any] # Action | State

    def step(self, action: Action):
        state = self.history[-1]
        assert isinstance(state, State)

        # Sanity check invalid action
        feature, thresh = action
        assert feature == 0 # XXX: Only single feature right now
        assert 0.0 <= thresh <= 1.0

        # Add next (a,s) pair to history
        state_p = state.clone()
        state_p.tree.split(
            node_id=0, # Root
            feature=feature,
            thresh=thresh
        )
        self.history.append(action)
        self.history.append(state_p)

    def reward(self) -> float:
        # accuracy + penalty for train/infer time
        # - accuracy ~ logloss
        tree = self.history[-1].tree
        return Evaluation.evaluate_tree(tree, self.dataset)


@dataclass
class Env:
    dataset: Any
    max_depth: int

    def spawn(self) -> Episode:
        n_features = len(self.dataset[0][0])
        n_outcomes = 2 # XXX: Make not fixed
        tree = Tree.dense(max_depth=1, n_outcomes=n_outcomes)
        initial_state = State(
            n_features=n_features,
            tree=tree,
        )
        return Episode(
            dataset=self.dataset,
            history=[initial_state],
        )

    def encoded_state_size(self) -> int:
        ep = self.spawn()
        return len(ep.current().encode())

    def encoded_action_size(self) -> int:
        return THRESH_SIZE

    def to_action_idx(self, action: Action) -> int:
        feature, thresh = action
        thresh_idx = quantize(thresh, buckets=THRESH_SIZE)
        return thresh_idx

    def to_action(self, index: int) -> Action:
        # Samples a threshold uniformly from the bucket range
        feature = 0
        thresh = discretize(index, buckets=THRESH_SIZE)
        b_size = bucket_size(0.0, 1.0, THRESH_SIZE)
        thresh += random() * b_size
        return (feature, thresh)


class Datasets:

    @classmethod
    def simple(Cls, n_features=1, noise=0.10):
        '''
        Create a dataset w/ :n_features input features, one of which
        causes the label value.

        :noise controls P(AssignRandomLabel).
        '''
        inputs = [] # List[inputs]
        labels = [] # List[label_idx]
        causal_feat = choice(list(range(n_features)))
        for _ in range(1000):
            inp = []
            for i in range(n_features):
                f_val = random()
                inp.append(f_val)
            if random() <= noise:
                label = round(random()) # noise
            else:
                label = 1 if inp[causal_feat] <= 0.50 else 0
            inputs.append(inp)
            labels.append(label)
        return inputs, labels


class Models:

    @classmethod
    def single_split(Cls, thresh=0.50):
        tree = Tree.dense(max_depth=2, n_outcomes=2)
        tree.split(node_id=0, feature=0, thresh=thresh)
        return tree


class Tasks:

    def inspect_tree(self):
        tree = Models.single_split(thresh=0.50)
        print("\nTree")
        for node in tree.nodes:
            print(node)

        print("\nBFS")
        for node in tree.bfs():
            print(node)
        print()

    def check_evaluate(self):
        # Generate dataset
        inputs, labels = Datasets.simple(n_features=1, noise=0.10)

        # Generate tree + enrich outcomes
        tree = Models.single_split(thresh=0.50)
        tree.enrich_outcomes(inputs, labels)
        print(tree)

        # Evaluate tree
        for inp in ([0.1], [0.9]):
            est = tree.estimate(inp)
            print(inp, est)
        print()

    def check_env(self):

        def disp_step(ep):
            print("\nStep")
            cstate = ep.current()
            for node in cstate.tree.bfs():
                print(node)
            print("state encoding", cstate.encode())
            print("f mask", cstate.f_mask())
            print("b mask", cstate.b_mask())
            print(ep.done())

        dataset = Datasets.simple(n_features=1, noise=0.10)

        print("\nDataset")
        for i in range(5):
            print(dataset[0][i], dataset[1][i])

        env = Env(
            dataset=dataset,
            max_depth=1,
        )
        episode = env.spawn()
        actions = [(0, 0.50)]
        disp_step(episode)
        for action in actions:
            episode.step(action)
            print()
            disp_step(episode)

        print("\nEpisode Reward:", episode.reward())

        print("\nSteps:")
        print("n steps:", episode.n_steps())
        for step in episode.steps():
            print(step)
        print()


if __name__ == "__main__":
    Tasks().inspect_tree()
    Tasks().check_evaluate()
    Tasks().check_env()
