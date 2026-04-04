"""
EORL: Energy Optimization via Reinforcement Learning in SDWSN
Implementation ตาม paper: Boonlert et al., InCIT2024

การแก้ไขจากโค้ดเดิม:
  1. Topology + distances ตาม Fig.3 ใน paper
  2. Energy model: first-order (Etx/Erx คิด distance) ตาม Eq.7-8
  3. Q-table: 2D [state][action] แทน 1D
  4. State function: st(min) หรือ st(max-min) ตาม Eq.2-3
  5. Reward: Rmin = min(ERi/Einit) ตาม Eq.6
  6. Lifetime = first bottleneck node (1/2/3) dies ตาม paper Fig.4
  7. Node death handling: ลบ routes ที่ผ่าน dead nodes
"""

import networkx as nx
import numpy as np
import random
import math
from itertools import combinations


# =========================
# PARAMETERS (Table I)
# =========================

Eelec      = 50e-9       # 50  nJ/bit
Efs        = 100e-12     # 100 pJ/bit/m²
Emp        = 0.0013e-12  # 0.0013 pJ/bit/m⁴
bl         = 100 * 8     # 100 bytes = 800 bits
Einit      = 5.0         # 5 J initial energy
ALPHA      = 0.5         # learning rate
GAMMA      = 0.5         # discount factor
EPSILON    = 0.1         # ε-greedy exploration
d0         = math.sqrt(Efs / Emp)   # ~277 m distance threshold

# State discretization: 0 → 5 J ทุก 0.01 J = 501 states (ตาม paper)
STATE_STEP = 0.01
N_STATES   = int(Einit / STATE_STEP) + 1   # 501

# Bottleneck nodes ตาม paper (nodes 1,2,3 เป็น relay หลัก)
BOTTLENECK = {1, 2, 3}
SINK       = 0


# =========================
# TOPOLOGY (Fig. 3)
# =========================

def build_topology():
    """
    9 nodes: node 0 = sink/controller, nodes 1-8 = sensor nodes
    edges พร้อม distance (เมตร) อ่านจาก Fig. 3
    """
    g = nx.Graph()
    g.add_nodes_from(range(9))

    edge_distances = {
        (0, 1): 75,
        (0, 2): 50,
        (0, 3): 90,
        (1, 2): 55,
        (1, 4): 37,
        (2, 3): 50,
        (2, 4): 62,
        (2, 6): 37,
        (3, 5): 62,
        (3, 6): 62,
        (4, 6): 55,
        (4, 7): 37,
        (5, 6): 74,
        (5, 8): 65,
        (6, 7): 67,
        (6, 8): 74,
        (7, 8): 65,
    }

    for (u, v), d in edge_distances.items():
        g.add_edge(u, v, distance=d)

    return g


# =========================
# ENERGY MODEL (Eq. 7-8)
# =========================

def etx(d):
    """Energy consumption for transmission (Joules)"""
    if d <= d0:
        return bl * (Efs * d**2 + Eelec)
    else:
        return bl * (Emp * d**4 + Eelec)


def erx():
    """Energy consumption for reception (Joules)"""
    return bl * Eelec


# =========================
# GENERATE SPANNING TREES
# =========================

def generate_spanning_trees(graph):
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    trees = []

    for subset in combinations(edges, len(nodes) - 1):
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(subset)

        if nx.is_tree(g):
            trees.append(g)

    return trees


# =========================
# ROUTING
# =========================

def tree_to_routes(tree, active_nodes, sink=SINK):
    """
    คำนวณ routes จาก spanning tree
    [FIX] ตรวจ path ผ่าน dead nodes และใส่ try/except
    """
    routes = {}

    for node in active_nodes:
        try:
            path = nx.shortest_path(tree, node, sink)
            # ตรวจว่าทุก relay ใน path ยังมีชีวิต
            if all(n in active_nodes or n == sink for n in path):
                routes[node] = path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    return routes


def sdwsn_routing(graph, active_nodes, sink=SINK):
    """
    SDWSN baseline: shortest path บน subgraph ของ active nodes
    (รองรับ dynamic topology เมื่อ node ตาย)
    """
    sub = graph.subgraph(list(active_nodes) + [sink])
    routes = {}

    for node in active_nodes:
        try:
            routes[node] = nx.shortest_path(sub, node, sink)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    return routes


# =========================
# SIMULATION
# =========================

def simulate_round(routes, energy, graph):
    """Simulate one round of data transmission"""
    for node, path in routes.items():
        for i in range(len(path) - 1):
            sender   = path[i]
            receiver = path[i + 1]
            d = graph[sender][receiver].get('distance', 50)
            energy[sender]   -= etx(d)
            energy[receiver] -= erx()


# =========================
# STATE FUNCTIONS (Eq. 2-3)
# =========================

def get_state_min(energy, active_nodes):
    """
    st(min) = discretized minimum remaining energy (Eq. 2)
    ใช้ติดตาม bottleneck node ที่พลังงานน้อยสุด
    """
    min_e = max(0.0, min(energy[n] for n in active_nodes))
    return min(int(min_e / STATE_STEP), N_STATES - 1)


def get_state_maxmin(energy, active_nodes):
    """
    st(max-min) = discretized (max - min) energy (Eq. 3)
    ใช้ติดตามความไม่สมดุลของพลังงาน
    """
    vals   = [energy[n] for n in active_nodes]
    maxmin = max(0.0, max(vals) - min(vals))
    return min(int(maxmin / STATE_STEP), N_STATES - 1)


# =========================
# REWARD FUNCTION (Eq. 6)
# =========================

def reward_min(energy, active_nodes):
    """
    Rmin(at) = min_i( ERi(st,at) / Einit )  -- Eq. 6
    คืนค่า normalized worst-case energy ของ node ที่แย่สุด
    """
    return min(max(0.0, energy[n]) / Einit for n in active_nodes)


# =========================
# SDWSN (baseline)
# =========================

def run_sdwsn(graph):
    """
    SDWSN baseline ด้วย shortest path + first-order energy model
    Lifetime = round ที่ bottleneck node แรกตาย (ตาม paper Fig.4)
    """
    energy = [Einit] * 9
    active = set(range(1, 9))
    rounds = 0

    while active:
        routes = sdwsn_routing(graph, active)
        if not routes:
            break

        simulate_round(routes, energy, graph)
        rounds += 1

        dead = {n for n in active if energy[n] <= 0}
        active -= dead

        # [FIX] Lifetime = first bottleneck node dies (ตาม paper)
        if dead & BOTTLENECK:
            break

    return rounds


# =========================
# EORL (Q-learning)
# =========================

class EORL:

    def __init__(self, n_trees):
        # [FIX] Q-table เป็น 2D: [state][action] ตาม paper
        self.Q       = np.zeros((N_STATES, n_trees))
        self.alpha   = ALPHA
        self.gamma   = GAMMA
        self.epsilon = EPSILON

    def choose_action(self, state):
        """ε-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.Q.shape[1] - 1)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action, reward, next_state):
        """Q-value update (Eq. 1)"""
        best_next = np.max(self.Q[next_state])
        self.Q[state, action] = (
            (1 - self.alpha) * self.Q[state, action]
            + self.alpha * (reward + self.gamma * best_next)
        )


def run_eorl(trees, graph, state_mode='min'):
    """
    EORL algorithm ตาม Algorithm 1 ใน paper
    state_mode: 'min' ใช้ st(min), 'maxmin' ใช้ st(max-min)
    Lifetime = round ที่ bottleneck node แรกตาย (ตาม paper Fig.4)
    """
    energy = [Einit] * 9
    active = set(range(1, 9))
    agent  = EORL(len(trees))
    rounds = 0

    get_state = get_state_min if state_mode == 'min' else get_state_maxmin

    while active:
        # คำนวณ current state
        s = get_state(energy, active)

        # เลือก action (spanning tree)
        action = agent.choose_action(s)
        routes = tree_to_routes(trees[action], active)

        # fallback ถ้า tree ที่เลือกไม่มี route ที่ใช้งานได้
        if not routes:
            routes = sdwsn_routing(graph, active)
        if not routes:
            break

        simulate_round(routes, energy, graph)
        rounds += 1

        # คำนวณ reward และ next state
        r      = reward_min(energy, active)
        dead   = {n for n in active if energy[n] <= 0}
        active -= dead
        s_next = get_state(energy, active) if active else 0

        # อัปเดต Q-table
        agent.update(s, action, r, s_next)

        # [FIX] Lifetime = first bottleneck node dies (ตาม paper)
        if dead & BOTTLENECK:
            break

    return rounds


# =========================
# MAIN
# =========================

def main():

    g = build_topology()

    print("Generating spanning trees...")
    trees = generate_spanning_trees(g)
    print(f"Total spanning trees: {len(trees)}")
    print()

    print("Running SDWSN simulation...")
    sdwsn_result = run_sdwsn(g)
    print(f"SDWSN lifetime: {sdwsn_result} rounds")
    print()

    print("Running EORL simulation [st(min), Rmin]...")
    eorl_min = run_eorl(trees, g, state_mode='min')
    print(f"EORL st(min) lifetime: {eorl_min} rounds")

    print("Running EORL simulation [st(max-min), Rmin]...")
    eorl_mm = run_eorl(trees, g, state_mode='maxmin')
    print(f"EORL st(max-min) lifetime: {eorl_mm} rounds")

    print()
    for name, val in [("EORL st(min)", eorl_min), ("EORL st(max-min)", eorl_mm)]:
        diff = val - sdwsn_result
        print(f"{name} vs SDWSN: {diff:+d} rounds ({diff/sdwsn_result*100:+.1f}%)")


if __name__ == "__main__":
    main()