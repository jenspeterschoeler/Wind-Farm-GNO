"""Top-level graph conversion module."""

from torch_geometric.data import Data as PyGData


class PyGTupleData(PyGData):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "global_features":
            return None
        if key == "n_node":
            return None
        if key == "n_edge":
            return None
        if key == "trunk_inputs":
            return None
        if key == "output_features":
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)
