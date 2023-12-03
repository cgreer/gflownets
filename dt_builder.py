from random import random, choice
from collections import deque
from dataclasses import dataclass
from functools import cached_property
from itertools import count
# import math
from typing import (
    Any,
    List,
    # Dict,
    Tuple,
    Optional,
)
from types import SimpleNamespace as Record

import torch

from partitions import Partition, Range, Endpoint

enu = enumerate

Tensor = torch.Tensor
NodeID = int
Action = Tuple[int, int, int, bool] # split_node_id, feature, thresh, stop
Splitting = Tuple[int, int, int] # NodeIdx, Feature, Threshold

THRESH_SIZE = 21


def bucket_size(min_value, max_value, buckets):
    return (max_value - min_value) / buckets


def quantize(value, min_value=0.0, max_value=1.0, buckets=THRESH_SIZE):
    # XXX: Double check this
    if value < min_value:
        return 0
    if value >= max_value:
        return buckets - 1
    b_size = bucket_size(min_value, max_value, buckets)
    return int((value - min_value) / b_size)


def discretize(bucket, min_value=0.0, max_value=1.0, buckets=THRESH_SIZE):
    # XXX: Make midpoint?
    b_size = bucket_size(min_value, max_value, buckets)
    return min_value + (bucket * b_size)


def max_splits(max_depth):
    '''
    Given a fully-split tree of depth :max_depth, what is the number
    of split nodes in that tree?"
    '''
    return 2**(max_depth) - 1


def create_feature_range(n_buckets):
    return Range(
        left=Endpoint(0, closed=True),
        right=Endpoint(n_buckets - 1, closed=True),
    )


@dataclass
class TreeNode:
    id: int
    depth: int
    parent: Optional[NodeID]
    outcomes: List[int]
    partition: Partition = None
    feature: Optional[int] = None
    thresh: Optional[float] = None
    pass_id: Optional[NodeID] = None
    fail_id: Optional[NodeID] = None

    def copy(self):
        partition = None
        if self.partition is not None:
            partition = self.partition.copy()
        return TreeNode(
            id=self.id,
            depth=self.depth,
            parent=self.parent,
            partition=partition,
            outcomes=self.outcomes[:],
            feature=self.feature,
            thresh=self.thresh,
            pass_id=self.pass_id,
            fail_id=self.fail_id,
        )

    def split_choices(self, feature):
        T = THRESH_SIZE

        # Feature hasn't been split
        # - Could be any t in T
        if feature not in self.partition:
            return list(range(T))

        # Feature has been split by ascendant node
        # - Get valid T choices
        else:
            t_range = self.partition[feature]

            lower = t_range.left.value
            assert isinstance(lower, int)
            if t_range.left.open:
                lower += 1

            higher = t_range.right.value
            assert isinstance(higher, int)
            if t_range.right.open:
                higher -= 1
            return list(range(lower, higher + 1))

    @property
    def split(self):
        return self.feature is not None


@dataclass
class Tree:
    max_depth: int # Depth is defined as n splits to reach leaf
    n_outcomes: int # n outcomes categorical being predicted has
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
            partition=Partition(),
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
        split_node = self.nodes[node_id]
        pass_id = split_node.pass_id
        fail_id = split_node.fail_id
        assert pass_id is not None
        assert fail_id is not None

        # Set split node's split info
        split_node.feature = feature
        split_node.thresh = thresh

        # Create child partitions
        # - Each child's partition is a subset of split node
        # - Split node will ALWAYS have a partiion
        assert split_node.partition is not None
        T = THRESH_SIZE
        child_ids = [pass_id, fail_id]
        child_types = [True, False] # am pass?
        for child_id, is_pass in zip(child_ids, child_types):
            child_part = split_node.partition.copy()
            if feature not in child_part:
                child_part[feature] = create_feature_range(T)
            if is_pass:
                child_part[feature].right.value = thresh # split threshold
                child_part[feature].right.closed = True # b/c <= is operation
            else:
                child_part[feature].left.value = thresh
                child_part[feature].left.closed = False # b/c > is operation
            self.nodes[child_id].partition = child_part

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

    def size(self):
        '''Size of active tree'''
        return len(self.bfs())

    def traverse(self, x):
        node = self.root
        while not self.leaf(node.id):
            threshold = discretize(
                node.thresh,
                min_value=0.0, # XXX: Change for real ranges
                max_value=1.0, # XXX: Change for real ranges
                buckets=THRESH_SIZE
            )
            if x[node.feature] <= threshold:
                node = self.nodes[node.pass_id]
            else:
                node = self.nodes[node.fail_id]
        return node

    def enrich_outcomes(self, inputs, labels):
        # XXX: make efficient via using only one pass / numpy ops
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

    def display(self):
        for node in self.bfs():
            feature = node.feature if node.feature is not None else ""
            thresh = node.thresh if node.thresh is not None else ""
            s = f"id:{node.id} f:{feature} t:{thresh}"
            print(" " * node.depth, s)


@dataclass
class Split:
    node_id: int = None # id of node to be split
    feature: int = None
    thresh: float = None

    def copy(self):
        return Split(
            node_id=self.node_id,
            feature=self.feature,
            thresh=self.thresh,
        )

    def complete(self):
        # is everything defined?
        if self.node_id is None:
            return False
        if self.feature is None:
            return False
        if self.thresh is None:
            return False
        return True

    def progress(self):
        s = ""
        if self.node_id is not None:
            s += "s"
        if self.feature is not None:
            s += "f"
        if self.thresh is not None:
            s += "t"
        assert s in ("", "s", "sf"), "Hierarchichal Issue"
        return s


@dataclass
class State:
    n_features: int
    tree: Tree
    split: Split # Working split
    stopped: bool

    def clone(self) -> 'State':
        return State(
            n_features=self.n_features,
            tree=self.tree.copy(),
            split=self.split.copy(),
            stopped=self.stopped,
        )

    def candidate_splits(self):
        '''
        List of nodes that are candidates to be split next.

        Must be:
        - A leaf node AND
        - Not deeper than max_depth AND
        - Must contain at least one feature with a threshold left to
          choose.
        '''
        # Get leaf nodes
        cands = []
        for node in self.tree.bfs(): # only selects active
            if not self.tree.leaf(node.id):
                continue
            if node.depth >= self.tree.max_depth:
                continue
            total_choices = 0
            for feat in range(self.n_features):
                t_choices = node.split_choices(feat)
                total_choices += len(t_choices)
            if total_choices <= 0:
                continue
            cands.append(node)
        return cands

    def frontier_splits(self):
        '''
        List of split nodes that are on the frontier of the tree.
        These are the split nodes that were last split (or root).

        A frontier node is one whose children are leaf nodes. I.e.,
        they are not inner split nodes.
        '''
        fnode_ids = set()
        for node in self.tree.bfs(): # only active
            if self.tree.leaf(node.id):
                if node.parent is None:
                    continue
                fnode_ids.add(node.parent)
        return [self.tree.nodes[x] for x in fnode_ids]

    def complete_tree(self):
        if self.split.node_id is not None:
            return False
        if self.split.feature is not None:
            return False
        if self.split.thresh is not None:
            return False
        return True

    def partial_tree(self):
        '''Are we working on split?'''
        return not self.complete_tree()

    def terminal(self) -> bool:
        # Took "stop" action, we're at valid tree and done.
        if self.stopped:
            return True

        # Nothing left to split
        if not self.candidate_splits():
            return True

        return False

    def valid_f_actions(self):
        '''
        Hierarchichal split assembly version.

        Possible states:
        - Stopped/Finished -> []
        - CompleteTree(None, None, None), not terminal -> stop + S
        - Working(S=s) -> F | S=s
        - Working(S=s, F=f) -> T | S=s, F=f
        '''
        valid = []

        # Stopped/Finished
        # Nothing to be done on terminal (stopped/finished) states
        # - should never be called here anyways...
        if self.terminal():
            return valid

        # CompleteTree(None, None, None)
        # - Stop
        # - Choose S | Tree
        if self.complete_tree():
            # If complete tree then allowed to stop
            valid.append((None, None, None, True))

            # Split choices
            if self.split.node_id is None:
                for cand in self.candidate_splits():
                    valid.append((cand.id, None, None, None))

        # Working(S=s) -> F | S=s
        # - Choose Feature | split
        elif self.split.feature is None:
            # XXX: Some are invalid? Used up all thresholds?
            split_node = self.tree.nodes[self.split.node_id]
            for feature in range(self.n_features):
                t_choices = split_node.split_choices(feature)
                if not t_choices:
                    continue
                valid.append((None, feature, None, None))

        # Working(S=s, F=f) -> T | S=s, F=f
        # - Choose Thresh | split, feature
        elif self.split.thresh is None:
            split_node = self.tree.nodes[self.split.node_id]
            t_choices = split_node.split_choices(self.split.feature)
            for t_idx in t_choices:
                valid.append((None, None, t_idx, None))
        else:
            raise RuntimeError("Not possible?")

        if not valid:
            breakpoint()

        assert valid
        return valid

    def valid_b_actions(self):
        '''
        Hierarchichal split assembly version.

        Possible states:
        - Stopped -> stop is only valid.
        - CompleteTree -> frontier split T's (likely multiple)
        - Working(S=s)-> return s
        - Working(S=s, F=f) -> return f
        '''
        if self.stopped:
            return [(None, None, None, True)]

        # CompleteTree (but not stopped)
        if self.complete_tree():
            valid = []
            for node in self.frontier_splits():
                # XXX: Is this issue with continuous value? ??
                valid.append((None, None, node.thresh, None))
            return valid

        # Working(S=s)-> return s
        # - Chose a node_id to split, undo it.
        elif self.split.progress() == "s":
            return [(self.split.node_id, None, None, None)]

        # Working(S=s, F=f) -> return f
        # - Chose a node_id to split, and feature. Undo feature.
        elif self.split.progress() == "sf":
            return [(None, self.split.feature, None, None)]
        else:
            raise RuntimeError("Not possible?")


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
        acc = acc / n_samples
        # penalty = math.log(tree.size())
        # score = acc - (.1 * penalty)
        return acc


@dataclass
class BaseEpisode:

    def current(self) -> State:
        s = self.history[-1]
        assert isinstance(s, State)
        return s

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
        state = self.current()
        assert sum([x is None for x in action]) == 3 # only 1 defined
        node_id, feature, thresh, stop = action
        if thresh is not None:
            assert isinstance(thresh, int)

        # Build next state
        state_p = state.clone()
        if stop:
            assert state.split.node_id is None
            assert state.split.feature is None
            assert state.split.thresh is None
            state_p.stopped = True
        else:
            # Update working split
            split = state.split.copy()
            if node_id is not None:
                split.node_id = node_id
            if feature is not None:
                split.feature = feature
            if thresh is not None:
                split.thresh = thresh

            # Add split if complete OR continue building split
            if split.complete():
                state_p.tree.split(
                    node_id=split.node_id,
                    feature=split.feature,
                    thresh=split.thresh,
                )
                state_p.split = Split()
            else:
                state_p.split = split
        self.history.extend([action, state_p])

    @cached_property
    def reward(self) -> float:
        '''
        Important this is a cached property!
        '''
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
        tree = Tree.dense(max_depth=self.max_depth, n_outcomes=n_outcomes)
        initial_state = State(
            n_features=n_features,
            tree=tree,
            split=Split(),
            stopped=False,
        )
        return Episode(
            dataset=self.dataset,
            history=[initial_state],
        )

    def n_features(self) -> int:
        return len(self.dataset[0][0])


@dataclass
class MLPCodec:
    env: Env

    def encoded_state_size(self) -> int:
        # Use initial state to get encoding size
        istate = self.env.spawn().current()
        return len(self.encode(istate))

    def encode(self, state) -> Tensor:
        '''
        Encode to NN representation

        For each node in tree:
        - feature ohe (or 0.0 if no split) ; F
        - thresh ohe (or 0.0 if no split) ; F
        - split indicator (1.0 if split else 0.0) ; 1
        - e.g. [feat1, thresh1, feat2, thresh2, split_ind] (per node)

        For the working "split":
        - split ohe (or 0.0 if not chosen) ; S
        - feature ohe (or 0.0 if not chosen) ; F
        - thresh value (or 0.0 if not chosen) ; 1
        - chose split indicator (1.0 if chosen else 0.0) ; 1
        - chose feature indicator (1.0 if chosen else 0.0) ; 1
        - chose thresh indicator (1.0 if chosen else 0.0) ; 1
        - Note: Since it's "working", we don't know which feature thresh belongs to
        - e.g. [split1, split2, split3, feat1, feat2, thresh, schosen, fchosen, tchosen]

        Sizes:
        - Each node's encoding: (2F + 1).
        - Working split encoding: (S + F + 4)
        - Total: N x (2F + 1) + (S + F + 4)
        '''
        S = max_splits(state.tree.max_depth)
        F = state.n_features
        encoding = []

        # Encode all nodes
        node_enc_size = (2 * F) + 1
        for node in state.tree.nodes: # ALL nodes in bfs order
            enc = [0.0] * node_enc_size
            if node.feature is not None:
                enc[node.feature * 2] = 1.0
                enc[node.feature * 2 + 1] = node.thresh
                enc[-1] = 1.0 # indicator
            encoding.extend(enc)

        # Encode working split
        split_enc = [0.0] * (S + F + 4)
        if state.split.node_id is not None:
            split_enc[state.split.node_id] = 1.0
            split_enc[-3] = 1.0 # indicator
        if state.split.feature is not None:
            split_enc[S + state.split.feature] = 1.0
            split_enc[-2] = 1.0 # indicator
        if state.split.thresh is not None:
            split_enc[S + F] = state.split.thresh
            split_enc[-1] = 1.0 # indicator
        encoding.extend(split_enc)

        return torch.tensor(encoding).float()

    def action_sizes(self):
        S = max_splits(self.env.max_depth)
        F = self.env.n_features()
        T = THRESH_SIZE
        return S, F, T

    def to_action_idx(self, action: Action) -> int:
        S, F, T = self.action_sizes()
        node_id, feature, thresh, stop = action
        if node_id is not None:
            return node_id
        if feature is not None:
            return S + feature
        elif thresh is not None:
            # thresh_idx = quantize(thresh, buckets=T)
            return S + F + thresh
        elif stop:
            return S + F + T
        else:
            raise RuntimeError("How possible?")

    def to_action(self, index: int) -> Action:
        S, F, T = self.action_sizes()
        node_id, feature, thresh, stop = None, None, None, None
        if index < S:
            node_id = index
        elif index < (S + F):
            feature = index - S
        elif index < (S + F + T):
            thresh = index - (S + F)
            # Below samples a threshold uniformly from the bucket range
            # XXX: Will require some code rearch to make use of this
            # thresh_index = index - (S + F)
            # thresh = discretize(thresh_index, buckets=T)
            # b_size = bucket_size(0.0, 1.0, T) # XXX: Replace w/ feature ranges
            # thresh += random() * b_size
        else:
            assert index == S + F + T
            stop = True
        return (node_id, feature, thresh, stop)

    def encoded_action_size(self) -> int:
        S, F, T = self.action_sizes()
        return S + F + T + 1 # +1 for stop

    def action_mask(self, state, forward: bool) -> Tensor:
        '''
        Action mask for backwards actions
        - allowed = 1, disallowed = 0
        '''
        S, F, T = self.action_sizes()
        A = S + F + T + 1
        mask = [0] * A
        actions = state.valid_f_actions() if forward else state.valid_b_actions()
        for action in actions:
            mask[self.to_action_idx(action)] = 1
        return torch.Tensor(mask).bool()

    def f_mask(self, state) -> Tensor:
        return self.action_mask(state, forward=True)

    def b_mask(self, state) -> Tensor:
        return self.action_mask(state, forward=False)


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
        print("CAUSAL FEATURE", causal_feat)
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

    @classmethod
    def random_tree(Cls, n_features, max_depth):
        # Dataset is only needed so that the environment can
        # understand n features. It doesn't influence the tree
        # created.
        dataset = ([[0] * n_features], [0])
        env = Env(
            dataset=dataset,
            max_depth=max_depth,
        )
        episode = env.spawn()
        while not episode.done():
            v_actions = episode.current().valid_f_actions()
            action = choice(v_actions)
            episode.step(action)
        return episode.current().tree


@dataclass
class HaystackTask:

    def generate(self, N, n_features, max_depth):
        # Generate tree
        # - Just keep trying until you get a :max_depth one...
        while True:
            tree = Models.random_tree(n_features, max_depth)
            md = max([x.depth for x in tree.bfs()])
            if md >= max_depth:
                break

        # Assign P(Outcome | LeafNode)
        for node in tree.bfs():
            if not tree.leaf(node.id):
                continue
            signal = 0.98
            noise = 1.0 - signal
            if random() <= 0.50:
                node.outcomes = [signal * N, noise * N]
            else:
                node.outcomes = [noise * N, signal * N]

        # Generate dataset w/ tree
        # - Generate random input (XXX: uniform?)
        # - Traverse tree and get label
        datasets = []
        for _ in range(2):
            inputs = [] # List[inputs]
            labels = [] # List[label_idx]
            for _ in range(N):
                x = []
                for _ in range(n_features):
                    x.append(random())

                P_Y = tree.estimate(x)
                label = 0 if random() <= P_Y[0] else 1
                inputs.append(x)
                labels.append(label)
            dataset = (inputs, labels)
            datasets.append(dataset)

        # Evaluate true tree
        test_acc = Evaluation.evaluate_tree(tree, datasets[1])

        task = Record()
        task.tree = tree
        task.train = datasets[0]
        task.test = datasets[1]
        task.test_acc = test_acc
        return task


class Tasks:

    def inspect_tree(self):
        tree = Models.single_split(thresh=0.50)
        print("\nTree")
        for node in tree.nodes:
            print(node)

        print("\nBFS")
        for node in tree.bfs():
            print(" " * node.depth, node)
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

    def inspect_mask(self, m, codec):
        # Show allowable actions
        for i, val in enu(m.tolist()):
            if val is False:
                continue
            action = codec.to_action(i)
            print("ALLOWED:", action)

    def pretty_mask(self, m):
        return ["T" if x is True else "F" for x in m.tolist()]

    def check_env(self):
        dataset = Datasets.simple(n_features=1, noise=0.10)
        env = Env(
            dataset=dataset,
            max_depth=2,
        )
        codec = MLPCodec(env=env)

        def disp_step(ep):
            action = None
            if len(ep.history) > 1:
                action = ep.history[-2]
            cstate = ep.current()
            for node in cstate.tree.bfs():
                print(node)
            encoding = codec.encode(cstate)
            fmask = codec.f_mask(cstate)
            bmask = codec.b_mask(cstate)

            print("\nSTEP")
            print("Action", action)
            print("state encoding", encoding.tolist())
            print("mask lens", len(fmask), len(bmask))
            print("f mask", self.pretty_mask(fmask))
            self.inspect_mask(fmask, codec)
            print("b mask", self.pretty_mask(bmask))
            self.inspect_mask(bmask, codec)
            print("done?", ep.done())

        print("\nDataset")
        for i in range(5):
            print(dataset[0][i], dataset[1][i])

        episode = env.spawn()
        actions = [
            (0, None, None, None), # split 0
            (None, 0, None, None), # feature 0
            (None, None, THRESH_SIZE // 2, None), # Middleish splitpoint
        ]
        disp_step(episode)
        for action in actions:
            episode.step(action)
            print()
            disp_step(episode)

        print("\nEpisode Reward:", episode.reward)

        print("\nSteps:")
        print("n steps:", episode.n_steps())
        for step in episode.steps():
            print(step)
        print()

    def check_rand_episodes(self):
        from timing import report_every
        dataset = Datasets.simple(n_features=5, noise=0.10)
        env = Env(
            dataset=dataset,
            max_depth=2,
        )
        for _ in range(100000):
            report_every("episode", 10000)
            episode = env.spawn()
            while not episode.done():
                v_actions = episode.current().valid_f_actions()
                action = choice(v_actions)
                episode.step(action)

    def check_boundaries(self):

        def split_points(ts):
            sps = []
            for t_idx in ts:
                point = discretize(t_idx, 0.0, 1.0, b)
                sps.append(point)
            return sps

        # range of 0.0 to 1.0
        # split into 3
        b = 3 # n buckets

        # Root
        T = [0, 1, 2]
        # thresh = 1 # <= 1
        print("Root", split_points(T))

        # Pass child
        T = [0, 1]
        print("PChild", split_points(T))

        # Fail child
        T = [2]
        print("FChild", split_points(T))

    def invalid_thresh_example(self):

        def split_points(ts, b):
            sps = []
            for t_idx in ts:
                point = discretize(t_idx, 0.0, 1.0, b)
                sps.append(point)
            return sps

        # range of 0.0 to 1.0
        b = THRESH_SIZE
        T = list(range(b))
        print("SPs", split_points(T, b))

    def inspect_haystack(self, task):
        print("\nTree:")
        task.tree.display()
        print("\nPartitions:")
        for node in task.tree.bfs():
            if task.tree.leaf(node.id):
                print()
                print(node.partition)
                print(node.outcomes)
        print()

        n_inputs = len(task.train[0])
        tot_positive = sum(task.train[1])
        print("ratio", tot_positive / n_inputs)

        print("True tree acc:", task.test_acc)

    def check_haystack(self):
        N = 2000
        haystack = HaystackTask()
        task = haystack.generate(
            N=N,
            n_features=10,
            max_depth=3,
        )
        self.inspect_haystack(task)


if __name__ == "__main__":
    # Tasks().inspect_tree()
    # Tasks().check_evaluate()
    # Tasks().check_env()
    # Tasks().check_rand_episodes()
    # Tasks().invalid_thresh_example()
    # Tasks().check_boundaries()
    Tasks().check_haystack()
