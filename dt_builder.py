import math
from random import random, choice, shuffle
from collections import deque, defaultdict
from dataclasses import dataclass
from functools import cached_property
from itertools import count
# import math
from typing import (
    Any,
    List,
    Tuple,
    Optional,
)
from types import SimpleNamespace as Record

import numpy as np
import torch
from rich import print as rprint

from partitions import Partition, Range, Endpoint
from timing import report_every

enu = enumerate

Tensor = torch.Tensor
Matrix2D = Any # Numpy array
Array = Any # Numpy array (1D)
Mask = Any # Numpy array (bool typed)
NodeID = int
Action = Tuple[int, int, int, bool] # split_node_id, feature, thresh, stop
Splitting = Tuple[int, int, int] # NodeIdx, Feature, Threshold

THRESH_SIZE = 21

Prob = float # Type to indicate expects to sum to 1


class Calculations:

    @staticmethod
    def entropy(probs: List[Prob], eps=1e-10):
        # XXX: Which base?
        ent = 0.0
        for prob in probs:
            prob = max(prob, eps)
            ent += (prob * math.log2(prob))
        return -ent

    @staticmethod
    def info_gain(original: float, final: float):
        '''
        :original - Entropy of P(X)
        :final - Entropy of P(X | ...)

        Note that gains are positive! That's why it is subtracting
        original - final. IG reports how much information was gained,
        not the raw entropy decrease from prior to updated.
        '''
        return original - final


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
    selections: Mask = None
    outcomes: Array = None # counts of label outcomes
    partition: Partition = None
    feature: Optional[int] = None
    thresh: Optional[float] = None
    pass_id: Optional[NodeID] = None
    fail_id: Optional[NodeID] = None

    def copy(self):
        partition = None
        if self.partition is not None:
            partition = self.partition.copy()
        outcomes = None
        if self.outcomes is not None:
            outcomes = self.outcomes.copy()
        selections = None
        if self.selections is not None:
            selections = self.selections.copy()
        return TreeNode(
            id=self.id,
            depth=self.depth,
            parent=self.parent,
            selections=selections,
            outcomes=outcomes,
            partition=partition,
            feature=self.feature,
            thresh=self.thresh,
            pass_id=self.pass_id,
            fail_id=self.fail_id,
        )

    def size(self):
        return self.selections.sum()

    def split_choices(self, feature):
        '''
        Lower part of range can be either open/closed. Higher part
        of range will always be closed:

            |-----|
                  o-------|
                          o-------|

        '''
        T = THRESH_SIZE

        # Feature hasn't been split
        # - Could be any valid threshold
        if feature not in self.partition:
            lower = 1
            higher = T - 2

        # Feature has been split by ascendant node
        # - Get valid T choices
        else:
            t_range = self.partition[feature]
            lower = t_range.left.value + 1
            higher = t_range.right.value - 1
        assert isinstance(lower, int)
        assert isinstance(higher, int)
        return list(range(lower, higher + 1))

    @property
    def split(self):
        return self.feature is not None


@dataclass
class Tree:
    max_depth: int # Depth is defined as n splits to reach leaf
    n_outcomes: int # n outcomes categorical being predicted has
    nodes: List[TreeNode]
    observations: Matrix2D
    labels: Array

    @property
    def root(self):
        return self.nodes[0]

    @classmethod
    def dense(
        Cls,
        max_depth: int,
        n_outcomes: int,
        observations: Matrix2D,
        labels: Array,
    ):
        id_counter = count()

        # Seed bfs expansion
        queue = deque()
        root_sel = np.array([True] * len(observations))
        root = TreeNode(
            id=next(id_counter),
            depth=0,
            parent=None,
            selections=root_sel,
            outcomes=Cls.outcome_counts(n_outcomes, labels, root_sel),
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
            )
            fchild = TreeNode(
                id=next(id_counter),
                depth=node.depth + 1,
                parent=node.id,
            )
            node.pass_id = pchild.id
            node.fail_id = fchild.id
            queue.append(pchild)
            queue.append(fchild)

        tree = Tree(
            max_depth=max_depth,
            n_outcomes=n_outcomes,
            nodes=nodes,
            observations=observations,
            labels=labels,
        )
        return tree

    @classmethod
    def outcome_counts(Cls, n_outcomes, labels, selection):
        counts = np.zeros(n_outcomes, 'int')
        uniq, cnt = np.unique(labels[selection], return_counts=True)
        counts[uniq] = cnt
        return counts

    def copy(self):
        nodes = [x.copy() for x in self.nodes]
        return Tree(
            max_depth=self.max_depth,
            n_outcomes=self.n_outcomes,
            nodes=nodes,
            observations=self.observations, # XXX: NO COPY?
            labels=self.labels, # XXX: NO COPY?
        )

    def split(self, node_id: NodeID, feature: int, thresh: float):
        split_node = self.nodes[node_id]
        assert split_node.pass_id is not None
        assert split_node.fail_id is not None
        pass_node = self.nodes[split_node.pass_id]
        fail_node = self.nodes[split_node.fail_id]

        # Set split node's split info
        split_node.feature = feature
        split_node.thresh = thresh # bucket

        # Partition observations/labels
        # - Get observations that would pass just this split. And then
        #   AND those observations together with observations already
        #   selected by this node (before any splitting happened).
        thresh_val = discretize(
            thresh,
            min_value=0.0, # XXX: Change for real ranges
            max_value=1.0, # XXX: Change for real ranges
            buckets=THRESH_SIZE
        )
        passed_split = self.observations[:, feature] <= thresh_val
        pass_node.selections = split_node.selections & passed_split
        fail_node.selections = split_node.selections & ~passed_split

        # Set child outcomes
        pass_node.outcomes = self.outcome_counts(self.n_outcomes, self.labels, pass_node.selections)
        fail_node.outcomes = self.outcome_counts(self.n_outcomes, self.labels, fail_node.selections)

        # Create child partitions
        # - Each child's partition is a subset of split node
        # - Split node will ALWAYS have a partition
        assert split_node.partition is not None
        T = THRESH_SIZE
        child_nodes = [pass_node, fail_node]
        child_types = [True, False] # am pass?
        for child_node, is_pass in zip(child_nodes, child_types):
            child_part = split_node.partition.copy()
            if feature not in child_part:
                child_part[feature] = create_feature_range(T)
            if is_pass:
                child_part[feature].right.value = thresh # split threshold
                child_part[feature].right.closed = True # b/c <= is operation
            else:
                child_part[feature].left.value = thresh
                child_part[feature].left.closed = False # b/c > is operation
            child_node.partition = child_part

    def unsplit(self, node_id: NodeID):
        split_node = self.nodes[node_id]
        pass_node = self.nodes[split_node.pass_id]
        fail_node = self.nodes[split_node.fail_id]

        # Can't unsplit a node that has children that are split
        # nodes...
        assert self.leaf(split_node.pass_id), f"{split_node.pass_id} not leaf"

        # Unset split node's split info
        split_node.feature = None
        split_node.thresh = None

        # Unset child info
        for node in (pass_node, fail_node):
            node.selections = None
            node.outcomes = None
            node.partition = None

    def gain(self, node_id: NodeID, weighted=False):
        '''
        :weighted - If true, weight the info gain of pre/post split by
        the size of the node. I.e., the # of samples that are present
        in the dataset this tree is training on.
        '''
        # Gather info necessary for computation
        node = self.nodes[node_id]
        node_size = sum(node.outcomes)
        assert node.split
        assert node_size > 0
        assert len(node.outcomes) == 2, "Not supported yet"

        pass_node = self.nodes[node.pass_id]
        pass_size = sum(pass_node.outcomes)
        pass_weight = pass_size / node_size
        pass_ent = self.entropy(pass_node.id)
        assert pass_size > 0

        fail_node = self.nodes[node.fail_id]
        fail_size = sum(fail_node.outcomes)
        fail_weight = fail_size / node_size
        fail_ent = self.entropy(fail_node.id)
        assert fail_size > 0

        # Calculate final gain
        pre_split = self.entropy(node.id)
        post_split = (pass_weight * pass_ent) + (fail_weight * fail_ent)
        gain = Calculations.info_gain(pre_split, post_split)
        if weighted:
            gain = node_size * gain
        return gain

    def entropy(self, node_id: NodeID):
        node = self.nodes[node_id]
        total = sum(node.outcomes)
        normed = [x / total for x in node.outcomes]
        return Calculations.entropy(normed)

    def frontier_splits(self, unsplittable: bool):
        '''
        :unsplittable - If true, only count a node as "frontier" if it
        is one that you could unsplit. I.e., both children of that
        node must be a leaf and are not allowed to be a split node.
        '''
        fnode_ids = set()
        for node in self.bfs(): # only active
            if self.leaf(node.id):
                if node.parent is None:
                    continue
                fnode_ids.add(node.parent)

        # Final pass
        # - Check both children are leafs
        #   - It's possible that only one child is, and you can't
        #   unsplit a node where only one child is a leaf.
        fnodes = []
        for cand_id in fnode_ids:
            node = self.nodes[cand_id]
            if unsplittable:
                if not self.leaf(node.pass_id):
                    continue
                if not self.leaf(node.fail_id):
                    continue
            fnodes.append(node)
        return fnodes

    def prune(self, cutoff, max_pruned=1_000_000):
        for _ in range(max_pruned):
            did_prune = self.prune_one(cutoff=cutoff)
            if not did_prune:
                break

    def prune_one(self, cutoff):

        def should_prune(el):
            # Any child node have size 0
            for child_id in (el.pass_id, el.fail_id):
                if sum(self.nodes[child_id].outcomes) <= 0:
                    return True

            # Split gain isn't >= cutoff
            gain = self.gain(el.id) # Gain must exist now since children have data
            if gain < cutoff:
                return True
            return False

        for split_node in self.frontier_splits(True):
            if split_node.parent is None: # can't prune root
                continue
            if should_prune(split_node):
                self.unsplit(split_node.id)
                return True
        return False

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

    def leaves(self):
        lnodes = []
        for x in self.bfs():
            if self.leaf(x.id):
                lnodes.append(x)
        return lnodes

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

    def utilized_features(self):
        '''
        The unique # of features used across all split points.
        '''
        feats = set()
        for node in self.bfs():
            if not node.split:
                continue
            feats.add(node.feature)
        return feats

    def traversal(self, x):
        trav = [self.root]
        node = trav[-1]
        while not self.leaf(node.id):
            threshold = discretize(
                node.thresh,
                min_value=0.0, # XXX: Change for real ranges
                max_value=1.0, # XXX: Change for real ranges
                buckets=THRESH_SIZE
            )
            if x[node.feature] <= threshold:
                trav.append(self.nodes[node.pass_id])
            else:
                trav.append(self.nodes[node.fail_id])
            node = trav[-1]
        return trav

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

    def traverse_batch(self, X: Matrix2D):
        root_sel = np.array([True] * len(X))

        # Special case: no splits, only root
        if self.leaf(self.root.id):
            return [(self.root, root_sel)]

        # Assign rows to leaf nodes
        stack = [(self.root, root_sel)]
        results = []
        while stack:
            node, node_sel = stack.pop()
            thresh_val = discretize(node.thresh, min_value=0.0, max_value=1.0, buckets=THRESH_SIZE) # XXX: Change for real ranges
            passed_split = X[:, node.feature] <= thresh_val

            pass_node = self.nodes[node.pass_id]
            to_append = results if self.leaf(pass_node.id) else stack
            to_append.append((pass_node, node_sel & passed_split))

            fail_node = self.nodes[node.fail_id]
            to_append = results if self.leaf(fail_node.id) else stack
            to_append.append((fail_node, node_sel & ~passed_split))
        return results

    def reannotate(self, observations, labels):

        # Reset tree attributes
        self.observations = observations
        self.labels = labels

        # Reset root attributes
        root_sel = np.array([True] * len(observations))
        self.root.selections = root_sel
        self.root.outcomes = self.outcome_counts(
            self.n_outcomes,
            labels,
            root_sel,
        )

        # Go in BFS order and redo each split, which will redo
        # selections/outcomes/partitions/etc.
        for node in self.bfs():
            if not node.split:
                continue
            self.split(
                node_id=node.id,
                feature=node.feature,
                thresh=node.thresh,
            )

    def node_estimate(self, node):
        assert isinstance(node.outcomes, np.ndarray)
        if node.outcomes.sum() <= 0:
            # print("WARNING: 0-count node! Uniform estimate")
            n_outcomes = len(node.outcomes)
            est = [1.0 / n_outcomes] * n_outcomes
            est = np.array(est, 'float')
        else:
            est = node.outcomes / node.outcomes.sum()
        return est

    def estimate(self, x):
        # XXX: Dirichlet Smoothing?
        # - Hierarchichal updates or just one update?
        leaf = self.traverse(x)
        return self.node_estimate(leaf)

    def estimate_batch(self, X):
        # XXX: Dirichlet Smoothing?
        # - Hierarchichal updates or just one update?
        estimates = [] # [(sel, est)]
        for leaf, sel in self.traverse_batch(X):
            est = self.node_estimate(leaf)
            estimates.append((sel, est))
        return estimates

    def accuracy(self):
        '''
        Compute accuracy of full tree
        '''
        dataset = (self.observations, self.labels)
        return Evaluation.accuracy(self, dataset)

    def node_accuracy(self, node_id):
        '''
        Compute accuracy of data assigned to node :node_id
        '''
        obs, labels = self.node_dataset(node_id=node_id)
        P_label = self.node_estimate(self.nodes[node_id])
        est = np.argmax(P_label)
        n_correct = (labels == est).sum()
        acc = n_correct / len(obs)
        return acc

    def is_valid(self):
        '''
        Check:
        - Partitions' endpoints aren't ever identical
        - A split feature's entire range is represented in final partitions
        '''
        coverage = defaultdict(set) # feature: set([thresh, ...])
        for leaf in self.leaves():
            for feature in leaf.partition.ranges:
                thresh_range = leaf.partition.ranges[feature]
                left_val = thresh_range.left.value
                right_val = thresh_range.right.value
                if left_val == right_val:
                    return False
                for i in range(left_val, right_val + 1):
                    coverage[feature].add(i)
        for feature in coverage:
            if len(coverage[feature]) != THRESH_SIZE:
                return False
        return True

    def node_dataset(self, node_id):
        # Get obs/labels for just this node
        node = self.nodes[node_id]
        return (
            self.observations[node.selections],
            self.labels[node.selections],
        )

    def thresh_info(self, node_id, n_features: int):
        T = THRESH_SIZE
        node = self.nodes[node_id]
        infos = {}
        obs, labels = self.node_dataset(node_id)
        for feat in range(n_features):
            lower, higher = 0, T
            if feat in node.partition:
                t_range = node.partition[feat]
                lower, higher = t_range.left.value, t_range.right.value
            infos[feat] = []
            for thresh in range(T):
                active = False
                min_val = None
                max_val = None
                size = None
                n_pos = None
                ratio = None
                if lower <= thresh < higher:
                    active = True

                    # Select rows that match feat/thresh range
                    # XXX: Swap out with proper feature ranges
                    # XXX: Do proper open/closed ranges
                    min_val = discretize(thresh, min_value=0.0, max_value=1.0, buckets=T)
                    max_val = discretize(thresh + 1, min_value=0.0, max_value=1.0, buckets=T)
                    sel = (obs[:, feat] >= min_val) & (obs[:, feat] <= max_val)

                    # Calculate ratio
                    size = sel.sum()
                    n_pos = labels[sel].sum() # XXX: Make multiclass
                    if size > 0:
                        ratio = n_pos / size

                x = Record(
                    active=active,
                    min_val=min_val,
                    max_val=max_val,
                    size=size,
                    n_pos=n_pos,
                    ratio=ratio,
                )
                infos[feat].append(x)
        return infos

    def scatterplot(self, node_id):
        import seaborn
        import matplotlib.pyplot as pp
        import pandas as pd

        # Convert to dataframe (needed for seaborn)
        dataset = self.node_dataset(node_id)
        col_names = [str(i) for i in range(len(dataset[0][0]))]
        df_X = pd.DataFrame(dataset[0], columns=col_names)
        df_y = pd.DataFrame(dataset[1], columns=['Label'])
        df = pd.concat([df_X, df_y], axis=1)

        # plot
        seaborn.pairplot(df, hue="Label")
        pp.show()

    def display(self):
        # DFS hierarchy
        stack = [self.root]
        while stack:
            node = stack.pop()

            if self.leaf(node.id):
                s = f"id:{node.id}"
                s += " counts:" + str(node.outcomes)
                s += " part: " + str(node.partition)
                color = "yellow"
            else:
                s = f"id:{node.id} f:{node.feature} t:{node.thresh}"
                s += " counts:" + str(node.outcomes)
                color = "white"

            # Display node
            s = f"[{color}]{s}[/{color}]"
            rprint("  " * node.depth, s)

            # Continue DFS
            # - Backwards to do passchild first
            if self.leaf(node.id):
                continue
            stack.append(self.nodes[node.fail_id])
            stack.append(self.nodes[node.pass_id])

    def game_display(self, n_features):
        '''
        Info a human needs to play the "Build a DT" game.
        '''
        # Tree Info
        tree_acc = self.accuracy()
        n_nodes = self.size()
        n_ufeats = len(self.utilized_features())
        n_leaves = len(self.leaves())

        rprint("\n[bold green]Tree[/bold green]")
        print("accuracy:", round(tree_acc, 2))
        print("nodes:", n_nodes)
        print("leaves:", n_leaves)
        print("utilized features:", n_ufeats)
        self.display()

        # Node Info
        rprint("\n[bold green]Nodes[/bold green]")
        for node in self.bfs():
            # node_acc = self.node_accuracy(node_id)
            partition = node.partition
            leaf_tag = ""
            if self.leaf(node.id):
                leaf_tag = " [bold yellow]LEAF[/bold yellow]"

            rprint("\nNode: " + str(node.id) + leaf_tag)
            print("acc:", round(self.node_accuracy(node.id), 3))
            print("size:", node.size())
            print("partition:", partition)
            # self.scatterplot(node.id)

            tinfo = self.thresh_info(node.id, n_features)
            thresh_header = ""
            for i in range(THRESH_SIZE):
                el = str(i)
                thresh_header += el.ljust(3)
            rprint("T:".ljust(6) + thresh_header)
            for feat in tinfo:
                view_info = [(i, x.ratio) for i, x in enu(tinfo[feat])]
                view = FeatureRatioView.ratio_view(view_info)
                rprint(f"f{feat}: ".ljust(6) + view)


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
            raise RuntimeError("How?")

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
            for node in self.tree.frontier_splits(True):
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
    def accuracy(Cls, tree, dataset):
        observations, labels = dataset
        estimates = tree.estimate_batch(observations)
        tot_correct = 0
        tot_est = 0
        for sel, P_label in estimates:
            est = np.argmax(P_label)
            tot_est += sel.sum()
            n_correct = (labels[sel] == est).sum()
            tot_correct += n_correct
        assert tot_est == len(observations)
        acc = tot_correct / len(observations)
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
        return Evaluation.accuracy(tree, self.dataset)


@dataclass
class Env:
    dataset: Any
    max_depth: int

    def spawn(self) -> Episode:
        n_features = len(self.dataset[0][0])
        n_outcomes = 2 # XXX: Make not fixed
        tree = Tree.dense(
            max_depth=self.max_depth,
            n_outcomes=n_outcomes,
            observations=self.dataset[0],
            labels=self.dataset[1],
        )
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

    def stop_action(self):
        return (None, None, None, True)

    @classmethod
    def pretty_action(Cls, action):
        if action[0] is not None:
            return f"Split node: {action[0]}"
        elif action[1] is not None:
            return f"Split feature: {action[1]}"
        elif action[2] is not None:
            thresh_val = discretize(action[2], 0.0, 1.0, THRESH_SIZE)
            return f"Split thresh: {action[2]} {thresh_val}"
        elif action[3] is not None:
            return "STOP"
        else:
            raise KeyError()

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
    def single_factor(Cls, n_features=1, noise=0.10, N=1000):
        '''
        Create a dataset w/ :n_features input features, one of which
        causes the label value.

        :noise controls P(AssignRandomLabel).
        '''
        observations = []
        labels = []
        causal_feat = choice(list(range(n_features)))
        # print("CAUSAL FEATURE", causal_feat)
        for _ in range(N):
            obs = []
            for i in range(n_features):
                f_val = random()
                obs.append(f_val)
            if random() <= noise:
                label = round(random()) # noise
            else:
                label = 1 if obs[causal_feat] <= 0.50 else 0
            observations.append(obs)
            labels.append(label)
        observations = np.array(observations, 'float')
        labels = np.array(labels, 'int')
        return observations, labels


class Models:

    @classmethod
    def single_split(Cls, observations, labels, thresh_val=0.50):
        tree = Tree.dense(
            max_depth=2,
            n_outcomes=2,
            observations=observations,
            labels=labels,
        )
        thresh = quantize(thresh_val)
        tree.split(node_id=0, feature=0, thresh=thresh)
        return tree

    @classmethod
    def fixed(Cls, observations, labels):
        '''
        Make a tree / a single split that will always be the same.

        Useful for testing.
        '''
        tree = Cls.single_split(observations, labels, thresh_val=0.50)
        return tree

    @classmethod
    def random_tree(
        Cls,
        n_features,
        max_depth,
        n_leaves=None,
        dataset=None,
    ):
        # Build a dummy dataset if one isn't specified
        if dataset is None:
            dataset = Datasets.single_factor(
                n_features=n_features,
                noise=0.10,
                N=1,
            )

        # Build tree
        env = Env(
            dataset=dataset,
            max_depth=max_depth,
        )
        episode = env.spawn()
        stop_action = env.stop_action()
        while not episode.done():
            v_actions = episode.current().valid_f_actions()

            # Should stop now?
            # - Force stop if at n_leaves
            # - Forbid stop if not at n_leaves
            force_stop = False
            if (n_leaves is not None) and (stop_action in v_actions):
                if len(episode.current().tree.leaves()) == n_leaves:
                    force_stop = True
                else:
                    assert len(v_actions) > 1
                    v_actions.remove(stop_action)

            # Take next step
            if force_stop:
                action = stop_action
            else:
                action = choice(v_actions)
            # print("action:", action)
            episode.step(action)
        return episode.current().tree


@dataclass
class HaystackTask:
    N: int
    n_partitions: int
    n_features: int
    max_depth: int
    tree: Any
    train: Any
    test: Any
    test_acc: float

    @classmethod
    def generate(
        Cls,
        N,
        n_partitions,
        n_features,
        max_depth,
    ):
        # Generate tree
        # - Just keep trying until you get one w/ right number of
        # partitions.
        # - XXX: Better way?
        while True:
            tree = Models.random_tree(n_features, max_depth, n_leaves=n_partitions)
            n_parts = len(tree.leaves())
            if n_parts != n_partitions:
                print("Couldn't gen n leaf tree", n_partitions, n_parts)
                continue

            # Assign P(Outcome | LeafNode)
            # - Assign low entropy outcomes to leaf nodes to simulate
            #   a "good" tree that could be trained on a dataset that
            #   has high accuracy. The so-called "needle" in this
            #   haystack.
            # - A high gain split is created for each frontier node.
            #   I.e., the pass partition is high P(A) and fail
            #   partition is high P(not A).
            leaf_outcomes = {}
            for node in tree.frontier_splits(unsplittable=False):
                child_outcomes = []
                for i in range(2):
                    signal = 0.95 + .03 * random()
                    noise = 1.0 - signal
                    sig_count = int(signal * N)
                    noise_count = int(noise * N)
                    if i == 0:
                        outcomes = [sig_count, noise_count]
                    else:
                        outcomes = [noise_count, sig_count]
                    child_outcomes.append(outcomes)
                shuffle(child_outcomes)
                leaf_outcomes[node.pass_id] = child_outcomes[0]
                leaf_outcomes[node.fail_id] = child_outcomes[1]

            # Generate train/test datasets w/ tree
            # - Generate random input (XXX: uniform?)
            # - Traverse tree to get leaf node
            # - Use above outcomes to generate labels
            datasets = []
            for _ in range(2):
                observations = []
                labels = []
                for _ in range(N):
                    x = []
                    for _ in range(n_features):
                        x.append(random())
                    # P_Y = tree.estimate(x)
                    leaf = tree.traverse(x)
                    pseudo_outcomes = leaf_outcomes[leaf.id]
                    tot_samples = sum(pseudo_outcomes)
                    P_Y = [x / tot_samples for x in pseudo_outcomes]
                    label = 0 if random() <= P_Y[0] else 1
                    observations.append(x)
                    labels.append(label)
                observations = np.array(observations, 'float')
                labels = np.array(labels, 'int')
                dataset = (observations, labels)
                datasets.append(dataset)

            # Redo selections / outcomes w/ train set
            tree.reannotate(
                observations=datasets[0][0],
                labels=datasets[0][1],
            )

            # Prune
            # - Only zero-count partitions should be affected...
            # - Check that nothing was pruned, b/c this would mean
            # n_partitions is no longer correct.
            tree.prune(cutoff=0.01, max_pruned=1)
            if len(tree.leaves()) != n_partitions:
                continue

            # Check every leaf node has counts
            # - This is a rare exception. If you have a "side leaf" (a
            # leaf whose sibling is a split node), then that side leaf
            # is inelible for pruning. So if there isn't enough data
            # in the training dataset, some leafs will have no label
            # count data, which could mess up estimate()
            has_zero_nodes = False
            for node in tree.leaves():
                if sum(node.outcomes) <= 0:
                    has_zero_nodes = True
                    break
            if has_zero_nodes:
                continue

            # Success!
            break

        # Evaluate true tree
        test_acc = Evaluation.accuracy(tree, datasets[1])

        return Cls(
            N=N,
            n_partitions=n_partitions,
            n_features=n_features,
            max_depth=max_depth,
            tree=tree,
            train=datasets[0],
            test=datasets[1],
            test_acc=test_acc,
        )

    @classmethod
    def to_dataframe(self, dataset):
        import pandas as pd

        # Convert to dataframe
        col_names = [str(i) for i in range(len(dataset[0][0]))]
        df_X_train = pd.DataFrame(dataset[0], columns=col_names)
        df_y_train = pd.DataFrame(dataset[1], columns=['Label'])
        return pd.concat([df_X_train, df_y_train], axis=1)

    def scatterplot(self):
        import seaborn
        import matplotlib.pyplot as pp

        # Convert to dataframe
        df = self.to_dataframe(self.train)
        seaborn.pairplot(df, hue="Label")
        pp.show()

    def display(self):
        print("\nTree:")
        self.tree.display()

        print("\nSplits:")
        for node in self.tree.bfs():
            if not node.split:
                continue
            print("info gain:", self.tree.gain(node.id))

        print("\nPartitions:")
        for node in self.tree.bfs():
            if self.tree.leaf(node.id):
                print()
                print("partition:", node.partition)
                print("outcomes:", node.outcomes)
                print("entropy:", self.tree.entropy(node.id))
                if node.parent is not None:
                    print("parent gain:", self.tree.gain(node.parent))
        print()

        n_inputs = len(self.train[0])
        tot_positive = sum(self.train[1])
        print("ratio", tot_positive / n_inputs)

        print("True tree acc:", self.test_acc)


class FeatureRatioView:

    @classmethod
    def to_color(Cls, val):
        if val is None:
            return "#555555"
        red = "00"
        green = "00"
        if val <= 0.50:
            rval = 0.50 - val
            red = hex(round(255 * rval))[-2:]
        else:
            gval = val - 0.50
            green = hex(round(255 * gval))[-2:]
        return "#" + red + green + "00"

    @classmethod
    def ratio_view(Cls, vals):
        view = ""
        for thresh, val in vals:
            if val is None:
                sval = "  "
            else:
                val = min(val, 0.99)
                sval = str(round(val * 100))
                if len(sval) < 2:
                    sval = "0" + sval
            col = Cls.to_color(val)
            cell = f"[white on {col}]{sval} [/white on {col}]"
            view += cell
        return view


class Tasks:

    def single_split_setup(self):
        # Generate dataset
        observations, labels = Datasets.single_factor(n_features=1, noise=0.10)

        # Generate tree
        tree = Models.single_split(observations, labels, thresh_val=0.50)

        return (tree, observations, labels)

    def check_evaluate(self):
        tree, observations, labels = self.single_split_setup()
        for inp in ([0.1], [0.9]):
            est = tree.estimate(inp)
            print(inp, est)
        print()

    def check_evaluate_batch(self):
        tree, observations, labels = self.single_split_setup()
        estimates = tree.estimate_batch(observations)
        for sel, est in estimates:
            print(sel.sum(), est)

        # Get accuracy
        dataset = (observations, labels)
        acc = Evaluation.evaluate_tree_batch(tree, dataset)
        print("Acc:", acc)

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
        dataset = Datasets.single_factor(n_features=1, noise=0.10)
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
        dataset = Datasets.single_factor(n_features=5, noise=0.10)
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

    def check_haystack(self):
        haystack = HaystackTask.generate(
            N=1000,
            n_partitions=5,
            n_features=4,
            max_depth=4,
        )
        haystack.display()
        haystack.scatterplot()

    def check_entropy(self):
        dists = (
            (0.5, 0.5),
            (1.0, 0.0),
        )
        for dist in dists:
            print(dist, Calculations.entropy(dist))

    def check_selection_logic(self):
        # Example matrix of feature values
        matrix = np.array([
            [0.5, 0.2, 0.3],
            [0.1, 0.5, 0.2],
            [0.4, 0.1, 0.22],
            [0.3, 0.3, 0.19]
        ])

        parent_sel = np.array([True, False, True, True])

        # Finding the indices of rows where the value in the 3rd column is <= 0.21
        passed_split_mask = matrix[:, 2] <= 0.21
        pass_sel = parent_sel & passed_split_mask
        fail_sel = parent_sel & ~pass_sel # ANDing w/ parent IS necessary!

        print(matrix)
        print("parent sel", parent_sel)
        print()
        print("pmask", passed_split_mask)
        print()
        print("psel", pass_sel)
        print("fsel", fail_sel)

    def check_outcomes_logic(self):
        n_outcomes = 2 # size of categorical

        labels = np.array([0, 1, 0, 0, 1, 0])
        sel = np.array([True, False, True, True, True, True])
        print("labels", labels)
        print("sel", sel)

        # All labels
        counts = np.zeros(n_outcomes, 'int')
        uniq, cnt = np.unique(labels, return_counts=True)
        counts[uniq] = cnt
        print()
        print("all counts", counts)

        # Selected labels
        counts = np.zeros(n_outcomes, 'int')
        uniq, cnt = np.unique(labels[sel], return_counts=True)
        counts[uniq] = cnt
        print()
        print("selected counts", counts)

    def check_haystack_speed(self):
        for _ in range(2000):
            report_every("haystack", 100)
            HaystackTask.generate(
                N=2500,
                n_partitions=4,
                n_features=4,
                max_depth=5,
            )

    def check_tree_display(self):
        tree = Models.random_tree(
            n_features=1,
            max_depth=3,
        )
        tree.display()

    def check_random_trees(self):
        n_features = 4
        max_depth = 3

        # Generate a dataset
        observations, labels = Datasets.single_factor(
            n_features=n_features,
            noise=0.10,
            N=3000,
        )
        dataset = (observations, labels)

        # Generate a bunch of random trees
        # - Make sure each is valid according to sanity checks
        # - Make sure acc measures match
        N = 2000
        print(f"\nChecking {N} random trees")
        for _ in range(N):
            report_every("trees", 100)
            tree = Models.random_tree(
                n_features=n_features,
                max_depth=max_depth,
                dataset=dataset,
            )
            # Tree sanity check
            if not tree.is_valid():
                print("Invalid Tree:")
                tree.display()
                raise RuntimeError()

        print("...Success!")

    def debug_random_tree(self):
        n_features = 1
        max_depth = 3

        def display_state(ep):
            print()
            print()
            state = ep.current()
            print("Working Split:", state.split)
            state.tree.display()
            print()
            for v_action in state.valid_f_actions():
                print("valid:", v_action)

        actions = [
            (0, None, None, None),
            (None, 0, None, None),
            (None, None, 18, None),
            (1, None, None, None),
            (None, 0, None, None),
            (None, None, 4, None),
            (2, None, None, None),
            (None, 0, None, None),
            (None, None, 20, None),
            (3, None, None, None),
            (None, 0, None, None),
        ]
        env = Env(
            dataset=Datasets.single_factor(
                n_features=n_features,
                noise=0.10,
                N=1,
            ),
            max_depth=max_depth,
        )
        episode = env.spawn()
        for i, action in enu(actions):
            display_state(episode)
            print("Chose", action)
            episode.step(action)
        display_state(episode)

        print()
        for t in range(THRESH_SIZE):
            thresh_val = discretize(
                t,
                min_value=0.0, # XXX: Change for real ranges
                max_value=1.0, # XXX: Change for real ranges
                buckets=THRESH_SIZE
            )
            print(t, thresh_val)

    def check_accuracy(self):
        observations = []
        for _ in range(3):
            observations.append([0.1])
        for _ in range(5):
            observations.append([0.9])
        labels = [0, 0, 0, 1, 1, 1, 1, 1]
        observations = np.array(observations, 'float')
        labels = np.array(labels, 'int')

        tree = Models.fixed(observations, labels)

        eval_dataset = (observations, labels)
        acc = Evaluation.accuracy(tree, eval_dataset)
        print(acc)

    def tree_inspection(self):
        # Generate a dataset/tree
        n_features = 2
        max_depth = 2
        force_leaves = 4 # force 4 leaves
        observations, labels = Datasets.single_factor(
            n_features=n_features,
            noise=0.10,
            N=3000,
        )
        dataset = (observations, labels)
        tree = Models.random_tree(
            n_features=n_features,
            max_depth=max_depth,
            n_leaves=force_leaves,
            dataset=dataset,
        )

        # Tree Info
        tree.game_display(n_features=n_features)

    def check_colors(self):
        rprint("[rgb(100, 0, 0)]hello[/rgb(100, 0, 0)]")
        red = "[white on #d70000]0 1 [/white on #d70000]"
        black = "[white on #000000]2 3 [/white on #000000]"
        green = "[white on #00d700]4 4 [/white on #00d700]"
        rprint(red + black + green)

        # Make a fake thresh label ratios
        x = [None] * 10
        x[5] = 0.0
        x[6] = 0.20
        x[7] = 0.40
        x[8] = 0.60
        x[9] = 0.80
        x[9] = 1.00
        print(x)
        print()

        # Make view
        ratios = list(enu(x))
        rprint(FeatureRatioView.ratio_view(ratios))


if __name__ == "__main__":
    # Tasks().check_evaluate()
    # Tasks().check_env()
    # Tasks().check_rand_episodes()
    # Tasks().invalid_thresh_example()
    # Tasks().check_boundaries()
    # Tasks().check_haystack()
    # Tasks().check_frontier()
    # Tasks().check_entropy()
    # Tasks().check_selection_logic()
    # Tasks().check_outcomes_logic()
    # Tasks().check_haystack_speed()
    # Tasks().check_tree_display()
    # Tasks().debug_random_tree()
    # Tasks().check_evaluate_batch()
    # Tasks().check_random_trees()
    # Tasks().test_accuracy()
    # Tasks().check_colors()
    Tasks().tree_inspection()
