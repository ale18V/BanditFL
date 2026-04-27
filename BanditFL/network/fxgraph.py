import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm, trange


def generate_connected_graph(n, e, seed=None):
    """
    Function to generate a random connected graph with n nodes and e edges.
    """
    if e < n - 1 or e > n * (n - 1) // 2:
        raise ValueError("Invalid number of edges for a simple connected graph.")

    # Generate a random spanning tree
    G = nx.generators.trees.random_tree(n, seed=seed)
    G = nx.Graph(G)  # Ensure it's an undirected graph

    # Add extra edges randomly until the edge budget is met
    existing_edges = set(G.edges())
    possible_edges = set(
        (i, j) for i in range(n) for j in range(i + 1, n)
    ) - existing_edges

    extra_edges_needed = e - (n - 1)
    random.seed(seed)
    extra_edges = random.sample(possible_edges, extra_edges_needed)
    G.add_edges_from(extra_edges)

    return G




def graph_byz_robust(G,byz):
    """
    Function to verify if a graph is Byzantine robust
    in the sense, no honest node has a Byzantine majority
    """
    corrupted_nodes = []
    is_robust = True
    for node in G.nodes():
        if node in byz:
            continue
        neighbors = list(G.neighbors(node))
        byz_neighbors = len([n for n in neighbors if n in byz])
        honest_neighbors = len(neighbors) - byz_neighbors + 1
        if byz_neighbors >= honest_neighbors:
            corrupted_nodes.append(node)
            is_robust = False
    return is_robust, corrupted_nodes

if __name__ == "__main__":
    """combinations = list(itertools.combinations(range(20), 3))

    n_exp = 1000
    n_success = 0
    for i in trange(n_exp):
        G = generate_connected_graph(20, 60, seed = i)
        for byz in combinations:
            is_robust, corrupted_nodes = graph_byz_robust(G, byz)
            if not is_robust:
                #print(f"Byzantine nodes: {byz}, Corrupted nodes: {corrupted_nodes}")
                n_success += 1
                break
    """
    combinations = list(itertools.combinations(range(30), 6))
    print(len(combinations))

    n_exp = 100
    n_success = 0
    for i in trange(n_exp):
        G = generate_connected_graph(30, 225, seed = i)
        for byz in combinations:
            is_robust, corrupted_nodes = graph_byz_robust(G, byz)
            if not is_robust:
                #print(f"Byzantine nodes: {byz}, Corrupted nodes: {corrupted_nodes}")
                n_success += 1
                break

    print(f"Number of successful experiments: {n_success} out of {n_exp}")
    
    """G = generate_connected_graph(30, 225, seed = 0)
    byz = np.random.choice(range(20), size=3, replace=False)
    node_colors = ['red' if node in byz else 'blue' for node in G.nodes()]
    nx.draw_networkx(G, node_color=node_colors)

    

    plt.savefig("graph_for_robustness.png")

    byz = np.random.choice(range(20), size = 3, replace=False)
    print(byz)

    print(graph_byz_robust(G, byz))"""