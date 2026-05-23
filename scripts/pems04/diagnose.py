from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


REPORTS = Path("reports/figures")
REPORTS.mkdir(parents=True, exist_ok=True)
STEPS_PER_DAY = 288

data = np.load("data/PEMS04/PEMS04.npz")["data"].astype(np.float32)
edges = pd.read_csv("data/PEMS04/PEMS04.csv").rename(columns={"cost": "distance"})
T, N, _ = data.shape
speed = data[:, :, 2]

train_end = int(T * 0.6)
v_free = np.percentile(speed[:train_end], 95.0, axis=0)
y = (speed < 0.6 * v_free[None, :]).astype(np.uint8)
positive_rate = float(y.mean())

n_full_days = T // STEPS_PER_DAY
y_trim = y[: n_full_days * STEPS_PER_DAY]
by_tod = y_trim.reshape(n_full_days, STEPS_PER_DAY, -1).mean(axis=(0, 2))

fig, ax = plt.subplots(figsize=(10, 4))
hours = np.arange(STEPS_PER_DAY) * 5 / 60
ax.plot(hours, by_tod, lw=1.5)
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))
ax.set_xlabel("Hour of day")
ax.set_ylabel("Fraction of sensors congested")
ax.set_title(f"PEMS04 — daily congestion pattern (positive rate = {positive_rate:.3f})")
ax.grid(alpha=0.3)
out = REPORTS / "pems04_daily_congestion.png"
fig.tight_layout()
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved → {out}")

G = nx.DiGraph()
for n in range(N):
    G.add_node(n)
distances = edges["distance"].to_numpy()
sigma = float(distances.std())
for _, row in edges.iterrows():
    d = float(row["distance"])
    w = float(np.exp(-(d ** 2) / (sigma ** 2)))
    G.add_edge(int(row["from"]), int(row["to"]), distance=d, weight=w)
G_un = G.to_undirected()

print("\n--- Graph basic stats ---")
print(f"nodes:               {G.number_of_nodes()}")
print(f"edges (directed):    {G.number_of_edges()}")
print(f"density:             {nx.density(G):.4f}")
print(f"weakly conn comps:   {nx.number_weakly_connected_components(G)}")
print(f"strongly conn comps: {nx.number_strongly_connected_components(G)}")
print(f"avg in-degree:       {sum(d for _, d in G.in_degree()) / G.number_of_nodes():.2f}")
print(f"avg out-degree:      {sum(d for _, d in G.out_degree()) / G.number_of_nodes():.2f}")
print(f"avg clustering (un): {nx.average_clustering(G_un):.4f}")
largest_wcc = max(nx.weakly_connected_components(G), key=len)
print(f"largest WCC size:    {len(largest_wcc)} / {G.number_of_nodes()}")

degs = [d for _, d in G.degree()]
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(degs, bins=range(0, max(degs) + 2), edgecolor="black")
ax.set_xlabel("Degree (in + out)")
ax.set_ylabel("Sensors")
ax.set_title("PEMS04 — degree distribution")
out = REPORTS / "pems04_degree_distribution.png"
fig.tight_layout()
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nsaved → {out}")

pos = nx.spring_layout(G_un, seed=42, iterations=200, weight="weight")
fig, ax = plt.subplots(figsize=(8, 8))
nx.draw_networkx_edges(G, pos, alpha=0.25, arrows=False, width=0.6, ax=ax)
nx.draw_networkx_nodes(G, pos, node_size=15, ax=ax)
ax.set_title("PEMS04 sensor network — spring layout")
ax.axis("off")
out = REPORTS / "pems04_network_layout.png"
fig.tight_layout()
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved → {out}")
