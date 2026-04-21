import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# -----------------------------
# Graph
# -----------------------------
edge_distances = {
    (0, 1): 75, (0, 2): 50, (0, 3): 90,
    (1, 2): 55, (1, 4): 37,
    (2, 3): 50, (2, 4): 62, (2, 6): 37,
    (3, 5): 62, (3, 6): 62,
    (4, 6): 55, (4, 7): 37,
    (5, 6): 74, (5, 8): 65,
    (6, 7): 67, (6, 8): 74,
    (7, 8): 65,
}

N_NODES = 9
SINK = 0
BOTTLENECK = {1, 2, 3}

# -----------------------------
# Energy
# -----------------------------
Eelec = 50e-9
Efs   = 100e-12
Emp   = 0.0013e-12
bl    = 800
Einit = 5.0

ALPHA   = 0.5
GAMMA   = 0.5
EPSILON = 0.1

d0 = math.sqrt(Efs / Emp)

STATE_STEP = 0.01
N_STATES = int(Einit / STATE_STEP) + 1  # 501

# -----------------------------
# Energy model
# -----------------------------
def energy_tx(d):
    if d < d0:
        return bl * (Eelec + Efs * d**2)
    else:
        return bl * (Eelec + Emp * d**4)

def energy_rx():
    return bl * Eelec

# -----------------------------
# Build adjacency
# -----------------------------
graph = defaultdict(list)
for (u, v), d in edge_distances.items():
    graph[u].append((v, d))
    graph[v].append((u, d))

# -----------------------------
# Generate spanning trees (DFS)
# -----------------------------
def generate_spanning_trees():
    trees = []

    def dfs(tree_edges, visited):
        if len(tree_edges) == N_NODES - 1:
            trees.append(tree_edges.copy())
            return

        for u in list(visited):
            for v, d in graph[u]:
                if v not in visited:
                    tree_edges.append(((u, v), d))
                    visited.add(v)

                    dfs(tree_edges, visited)

                    visited.remove(v)
                    tree_edges.pop()

    dfs([], {0})
    return trees

print("Generating spanning trees...")
spanning_trees = generate_spanning_trees()
print("Total spanning trees:", len(spanning_trees))

N_ACTIONS = len(spanning_trees)

# -----------------------------
# Simulation
# -----------------------------
def simulate_round(tree, energies):
    new_energy = energies.copy()

    for (u, v), d in tree:
        cost_tx = energy_tx(d)
        cost_rx = energy_rx()

        new_energy[u] -= cost_tx
        new_energy[v] -= cost_rx

    return new_energy

# -----------------------------
# Q-learning
# -----------------------------
Q = np.zeros((N_STATES, N_ACTIONS))

print("Training RL...")

for episode in range(20000):

    if episode % 1000 == 0:
        print("Episode:", episode)

    energies = np.full(N_NODES, Einit)

    while True:
        min_energy = energies.min()
        state = int(min_energy / STATE_STEP)
        state = min(state, N_STATES - 1)

        if random.random() < EPSILON:
            action = random.randint(0, N_ACTIONS - 1)
        else:
            action = np.argmax(Q[state])

        tree = spanning_trees[action]

        new_energies = simulate_round(tree, energies)
        new_min = new_energies.min()

        next_state = int(new_min / STATE_STEP)
        next_state = min(next_state, N_STATES - 1)

        reward = new_min  # Rmin

        Q[state, action] += ALPHA * (
            reward + GAMMA * np.max(Q[next_state]) - Q[state, action]
        )

        energies = new_energies

        if new_min <= 0:
            break

# -----------------------------
# Reduce table
# -----------------------------
Rmin_table = np.max(Q, axis=1)
policy = np.argmax(Q, axis=1)

print("\nRmin table (first 20 rows):")
print(Rmin_table[:20])

# -----------------------------
# Simulation with best policy
# -----------------------------
energies = np.full(N_NODES, Einit)

rounds = []
energy_history = {1: [], 2: [], 3: []}

print("\nSimulating final policy...")

for r in range(20000):

    state = int(energies.min() / STATE_STEP)
    state = min(state, N_STATES - 1)

    action = policy[state]
    tree = spanning_trees[action]

    energies = simulate_round(tree, energies)

    rounds.append(r)

    for b in BOTTLENECK:
        energy_history[b].append(max(energies[b], 0))

    if energies.min() <= 0:
        print("First node died at round:", r)
        break

# -----------------------------
# Plot
# -----------------------------
print("Plotting graph...")

for b in BOTTLENECK:
    plt.plot(rounds, energy_history[b], label=f'Node {b}')

plt.xlabel("Rounds (0-20000)")
plt.ylabel("Energy (0-5 J)")
plt.ylim(0, 5)
plt.title("Energy Decay (Bottleneck Nodes)")
plt.legend()
plt.grid()
plt.show()