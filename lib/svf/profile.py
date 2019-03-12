from pathlib import Path
import os
import logging
import numpy as np
from sn.io.graph.load import Loader

class Profile:
    root = Path("")
    dataset_dir = Path("")
    log_dir = Path("")
    name = "enron"
    log_name = "example.log"

    @classmethod
    def __get_roots(cls):
        root = Path.cwd().parent
        dataset_dir = root.joinpath('datasets')
        log_dir = root.joinpath('logs')
        return root, dataset_dir, log_dir

    @classmethod
    def set_profile(cls, name: str, log_name: str):
        cls.name = name
        cls.log_name = log_name
        cls.root, cls.dataset_dir, cls.log_dir = cls.__get_roots()
        return cls.dataset_dir, cls.log_dir, cls.name, cls.log_name

    @classmethod
    def get_profile(cls):
        dataset_dir = Path(cls.dataset_dir)
        g = Loader(dataset_dir, cls.name)
        # グラフの頂点数，辺数
        num_of_nodes, num_of_edges = g.size()
        # グラフの次元数
        dim_hd = g.dim_hd()[1]
        return num_of_nodes, num_of_edges, dim_hd

    @classmethod
    def get_labels(cls):
        dataset_dir = Path(cls.dataset_dir)
        g = Loader(dataset_dir, cls.name)
        num_of_nodes, _ = g.size()
        try:
            labels = g.attribute("label")
        except:
            labels = list(map(str, range(num_of_nodes)))
        return labels
