import torch

import networkx as nx

from tqdm import tqdm
from itertools import chain


def intra_community_edges(G, partition):
    return sum(G.subgraph(block).size() for block in partition)

def inter_community_edges(G, partition):
    aff = dict(chain.from_iterable(((v, block) for v in block)
                                for block in partition))
    return sum(1 for u, v in G.edges() if aff[u] != aff[v])

def inter_community_non_edges(G, partition):
    aff = dict(chain.from_iterable(((v, block) for v in block)
                                    for block in partition))

    return sum(1 for u, v in nx.non_edges(G) if aff[u] != aff[v])
    
def performance(G, partition):
    # Compute the number of intra-community edges and inter-community
    # edges.
    intra_edges = intra_community_edges(G, partition)
    inter_edges = inter_community_non_edges(G, partition)
    # Compute the number of edges in the complete graph (directed or
    # undirected, as it depends on `G`) on `n` nodes.
    #
    # (If `G` is an undirected graph, we divide by two since we have
    # double-counted each potential edge. We use integer division since
    # `total_pairs` is guaranteed to be even.)
    n = len(G)
    total_pairs = n * (n - 1)
    if not G.is_directed():
        total_pairs //= 2
    return (intra_edges + inter_edges) / total_pairs

def connectivity(G, partition):
    inter_edges = inter_community_edges(G, partition)
    inter_non_edges = inter_community_non_edges(G, partition)

    return inter_edges / inter_non_edges


def homophily(G, partition):
    verbose = True
    homophily_dict = dict()

    avg_h = 0.0
    for c, p in partition.items():
        subgraph = G.subgraph(p)
        h = 0.0
        for i in p: 
            if i in G:  # TODO: make nicer, make sure that all i are in G. Why shouldn't this be the case?
                graph_degree = G.out_degree[i]
                subgraph_degree = subgraph.out_degree[i]

                if graph_degree > 0:
                    h += subgraph_degree / graph_degree

        avg_h += h
        h /= len(p)

        if verbose:
            homophily_dict[c] = h
            
    if verbose:
        homophily_dict['global'] = avg_h / G.number_of_nodes()
        #homophily_dict['global'] = avg_h / len(partition)  # earlier solution averages over means
        return homophily_dict
    else:
        return avg_h / G.number_of_nodes()
        #return avg_h / len(partition)  # s.a.

