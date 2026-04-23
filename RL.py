import numpy as np
import networkx as nx
import random
import math
import matplotlib.pyplot as plt
from itertools import combinations

# ── 1. Parameters & Configuration ──
Eelec = 50e-9; Efs = 100e-12; Emp = 0.0013e-12
bl = 100 * 8; Einit = 5.0
alpha = 0.1; gamma = 0.5; epsilon = 0.1 # เริ่มต้น epsilon ที่ 0.1 หรือ 1.0 ก็ได้
STATE_STEP = 0.01
N_STATES = int(Einit / STATE_STEP) + 1  # 501 states
MAX_ROUNDS = 5000 # ปรับจำนวนรอบตามความเหมาะสมของเครื่อง

# ── 2. สร้าง Graph ──
edge_distances = {
    (0, 1): 75, (0, 2): 50, (0, 3): 90, (1, 2): 55, (1, 4): 37,
    (2, 3): 50, (2, 4): 62, (2, 6): 37, (3, 5): 62, (3, 6): 62,
    (4, 6): 55, (4, 7): 37, (5, 6): 74, (5, 8): 65, (6, 7): 67,
    (6, 8): 74, (7, 8): 65,
}
G = nx.Graph()
for (u, v), d in edge_distances.items():
    G.add_edge(u, v, distance=d)

# ── 3. สร้าง Spanning Trees (แทนที่ฟังก์ชันที่ไม่มีใน NetworkX) ──
def get_all_spanning_trees(graph):
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    trees = []
    # Spanning Tree ต้องมี N-1 edges
    for subset in combinations(edges, len(nodes) - 1):
        g_temp = nx.Graph()
        g_temp.add_nodes_from(nodes)
        g_temp.add_edges_from(subset)
        if nx.is_tree(g_temp):
            trees.append(g_temp)
    return trees

all_trees = get_all_spanning_trees(G)
N_TREES = len(all_trees)
print(f"พบ Spanning Trees ทั้งหมด: {N_TREES} รูปแบบ")

# ── 4. พลังงานและฟังก์ชันคำนวณ ──
d0 = math.sqrt(Efs / Emp)
def get_etx(d): return bl * (Efs * d**2 + Eelec) if d <= d0 else bl * (Emp * d**4 + Eelec)
def get_erx(): return bl * Eelec

def get_state(energy):
    # state = min energy (0.0 to 5.0) -> index 0-500
    v = min(energy[n] for n in range(1, 9)) # ไม่นับ sink (node 0)
    return min(int(v / STATE_STEP), N_STATES - 1)

# ── 5. Main Loop (Q-Learning) ──
Q = np.zeros((N_STATES, N_TREES))
energy = np.full(9, Einit)
energy_history = []

for rnd in range(MAX_ROUNDS):
    s = get_state(energy)
    
    # E-greedy selection
    if random.random() < epsilon:
        action = random.randint(0, N_TREES - 1)
    else:
        action = np.argmax(Q[s])
        
    # Simulate routing ใน Tree ที่เลือก
    tree = all_trees[action]
    for node in range(1, 9):
        path = nx.shortest_path(tree, source=node, target=0)
        for i in range(len(path) - 1):
            sender, receiver = path[i], path[i+1]
            dist = G[sender][receiver]['distance']
            energy[sender] -= get_etx(dist)
            energy[receiver] -= get_erx()
    
    # Reward
    r = min(energy[n] for n in range(1, 9)) / Einit
    
    # Update State
    s_next = get_state(energy)
    
    # Q-Update
    Q[s, action] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, action])
    
    energy_history.append(min(energy[1:]))

# ── 6. ผลลัพธ์ ──
# Q-Table 501x1 (ค่าดีที่สุดในแต่ละ state)
q_table_501x1 = np.max(Q, axis=1).reshape(-1, 1)
print("\n--- Q-Table (501x1) สำเร็จ ---")
print(q_table_501x1)

# Plot กราฟพลังงาน
plt.figure(figsize=(10, 5))
plt.plot(energy_history)
plt.title("Min Residual Energy vs Rounds")
plt.xlabel("Rounds")
plt.ylabel("Min Energy (J)")
plt.grid(True)
plt.show()