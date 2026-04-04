import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import random


# =========================
# CONFIGURATION
# =========================

NODE_SIZES = [10, 15, 20, 30]
AVG_DEGREES = [3, 4, 5, 6]

RUNS_PER_CASE = 30

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================
# GRAPH GENERATION
# =========================

def generate_graph(n, avg_degree):
    """
    Generate a random connected graph
    with approximate average degree
    """

    edges = int(n * avg_degree / 2)

    while True:
        G = nx.gnm_random_graph(n, edges)

        if nx.is_connected(G):
            return G


# =========================
# KIRCHHOFF MATRIX TREE
# =========================

def count_spanning_trees(G):
    """
    Count number of spanning trees
    using Kirchhoff Matrix Tree Theorem
    """

    L = nx.laplacian_matrix(G).toarray()

    L_minor = L[1:, 1:]

    det = np.linalg.det(L_minor)

    return round(abs(det))


# =========================
# SINGLE EXPERIMENT
# =========================

def run_single_experiment(nodes, degree):

    G = generate_graph(nodes, degree)

    start = time.perf_counter()

    spt = count_spanning_trees(G)

    end = time.perf_counter()

    elapsed = end - start

    return spt, elapsed


# =========================
# MAIN EXPERIMENT LOOP
# =========================

results = []

for n in NODE_SIZES:

    for d in AVG_DEGREES:

        print(f"Running experiment N={n}, Degree={d}")

        spt_list = []
        time_list = []

        for _ in range(RUNS_PER_CASE):

            spt, t = run_single_experiment(n, d)

            spt_list.append(spt)
            time_list.append(t)

        avg_spt = np.mean(spt_list)
        avg_time = np.mean(time_list)

        results.append({
            "nodes": n,
            "avg_degree": d,
            "avg_spt": avg_spt,
            "avg_time": avg_time
        })


# =========================
# SAVE RESULT
# =========================

df = pd.DataFrame(results)

print(df)

df.to_csv("spt_experiment_results.csv", index=False)


# =========================
# PLOT GRAPH
# =========================

plt.figure()

for d in AVG_DEGREES:

    subset = df[df["avg_degree"] == d]

    plt.plot(
        subset["nodes"],
        subset["avg_spt"],
        marker='o',
        label=f"degree={d}"
    )

plt.xlabel("Number of Nodes")
plt.ylabel("Average Number of SPT")
plt.title("Nodes vs Spanning Trees")
plt.legend()
plt.grid(True)
plt.show()


plt.figure()

for d in AVG_DEGREES:

    subset = df[df["avg_degree"] == d]

    plt.plot(
        subset["nodes"],
        subset["avg_time"],
        marker='o',
        label=f"degree={d}"
    )

plt.xlabel("Number of Nodes")
plt.ylabel("Computation Time (seconds)")
plt.title("Nodes vs Computation Time")
plt.legend()
plt.grid(True)
plt.show()