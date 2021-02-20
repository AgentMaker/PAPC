from collections import defaultdict
import scipy.spatial
import numpy as np


def get_split_dims(tree, max_depth=7):
    split_dims = defaultdict(list)
    tree_idxs = defaultdict(list)

    def _get_split_dims(tree, level=0, parent=None):
        if tree is None:
            tree = parent
        if level >= max_depth:
            indices = tree.indices
            n = 2**(max_depth - level)
            if len(indices) > n:
                inds = np.random.choice(range(len(indices)), n)
                indices = indices[inds]
            elif len(indices) < n:
                indices = np.concatenate([indices, indices[0:1].repeat(n - len(indices))])
            tree_idxs[level].append(indices)
            return indices
        indices = np.concatenate([
            _get_split_dims(tree.lesser, level=level + 1, parent=tree),
            _get_split_dims(tree.greater, level=level + 1, parent=tree)
        ])
        if level < max_depth:
            tree_idxs[level].append(indices)
            split_dim = tree.split_dim
            if split_dim == -1:
                split_dim = parent.split_dim if (parent.split_dim > -1) else 0
            split_dims[level].append(split_dim)
            split_dims[level].append(split_dim)

        return indices

    _get_split_dims(tree, level=0)

    tree_idxs = list(tree_idxs.values())
    split_dims = list(split_dims.values())

    split_dims = [np.array(item).astype(np.int64) for item in split_dims]
    tree_idxs = [np.stack(branch).astype(np.int64) for branch in tree_idxs]

    return split_dims, tree_idxs


def build_ClasKDTree(point_set, depth):
    kdtree = scipy.spatial.cKDTree(point_set, leafsize=1, balanced_tree=True)
    split_dims, tree_idxs = get_split_dims(kdtree.tree, max_depth=depth)
    tree = [np.take(point_set, indices=indices, axis=0) for indices in tree_idxs]

    return split_dims, tree

def build_SegKDTree(point_set, label_set, depth):
    kdtree = scipy.spatial.cKDTree(point_set, leafsize=1, balanced_tree=True)
    split_dims, tree_idxs = get_split_dims(kdtree.tree, max_depth=depth)
    point_tree = [np.take(point_set, indices=indices, axis=0) for indices in tree_idxs]
    label_tree = [np.take(label_set, indices=indices, axis=0) for indices in tree_idxs]

    return split_dims, point_tree, label_tree