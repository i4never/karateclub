import multiprocessing
import random

import tqdm
from toolz import partition_all


class RandomWalker:
    """
    Class to do fast first-order random walks.

    Args:
        walk_length (int): Number of random walks.
        walk_number (int): Number of nodes in truncated walk.
    """

    def __init__(self, walk_length, walk_number, workers=10):
        self.walk_length = walk_length
        self.walk_number = walk_number
        self.workers = workers
        self.walks = []
        self.graph = None

    def do_walk(self, node):
        """
        Doing a single truncated random walk from a source node.

        Arg types:
            * **node** *(int)* - The source node of the diffusion.

        Return types:
            * **walk** *(list of strings)* - A single truncated random walk.
        """
        walk = [node]
        for _ in range(self.walk_length - 1):
            n = walk[-1]
            nebs = [n for n in self.graph.neighbors(n) if n != node]
            weights = [self.graph.get_edge_data(n, neb)['weight'] for neb in nebs]
            if len(nebs) > 0:
                walk = walk + random.choices(nebs, weights)
            else:
                break
        walk = [str(w) for w in walk]
        return walk

    def do_batch_walks(self, nodes):
        batch_walks = list()
        for node in tqdm.tqdm(nodes):
            for _ in range(self.walk_number):
                walk_from_node = self.do_walk(node)
                batch_walks.append(walk_from_node)
        return batch_walks

    def do_walks(self, graph):
        """
        Doing a fixed number of truncated random walk from every node in the graph.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to run the random walks on.
        """
        self.walks = []
        self.graph = graph

        batches = [par for par in partition_all(int(self.graph.number_of_nodes() / self.workers) + 1, self.graph.nodes)]

        with multiprocessing.Pool(self.workers) as pool:
            seqs = pool.map(self.do_batch_walks, batches)
            return seqs
