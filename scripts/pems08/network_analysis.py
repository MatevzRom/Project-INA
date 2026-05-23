from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd


REPORTS = Path("reports/figures")
REPORTS.mkdir(parents=True, exist_ok=True)

data  = np.load("data/PEMS08/PEMS08.npz")["data"].astype(np.float32)
edges = pd.read_csv("data/PEMS08/PEMS08.csv").rename(columns={"cost": "distance"})
T, N, _ = data.shape

G         = nx.DiGraph()
for n in range(N):
    G.add_node(n)
distances = edges["distance"].to_numpy()
sigma     = float(distances.std())
for _, row in edges.iterrows():
    d = float(row["distance"])
    w = float(np.exp(-(d ** 2) / (sigma ** 2)))
    G.add_edge(int(row["from"]), int(row["to"]), distance=d, weight=w)

nodes = sorted(G.nodes())
G_un  = G.to_undirected()

print("Computing spring layout …")
pos = nx.spring_layout(G_un, seed=42, iterations=300)

print("Computing centralities …")
in_deg   = dict(G.in_degree())
out_deg  = dict(G.out_degree())
bet      = nx.betweenness_centrality(G, normalized=True, weight=None)
clo      = nx.closeness_centrality(G)
lwcc     = max(nx.connected_components(G_un), key=len)
eig_lwcc = nx.eigenvector_centrality_numpy(G_un.subgraph(lwcc), weight=None)
eig      = {n: eig_lwcc.get(n, 0.0) for n in G.nodes()}
pr       = nx.pagerank(G, alpha=0.85, weight=None)

cent = pd.DataFrame.from_dict({
    n: {
        "in_degree":   in_deg[n],
        "out_degree":  out_deg[n],
        "degree":      in_deg[n] + out_deg[n],
        "betweenness": bet[n],
        "closeness":   clo[n],
        "eigenvector": eig[n],
        "pagerank":    pr[n],
    }
    for n in nodes
}, orient="index")
cent.index.name = "node"
cent.to_csv(Path("reports") / "pems08_centralities.csv")
print("centralities saved → reports/pems08_centralities.csv")

print("\n--- Centrality summary ---")
print(cent.describe().to_string())

print("\nTop-10 nodes by betweenness centrality:")
print(cent["betweenness"].sort_values(ascending=False).head(10).to_string())

print("\nDetecting communities …")
partition = community_louvain.best_partition(G_un, random_state=42)
comm      = np.array([partition[n] for n in nodes], dtype=np.int32)
n_comm    = comm.max() + 1
print(f"Louvain: {n_comm} communities")

comm_df = pd.DataFrame({"node": nodes, "community": comm})
comm_df.to_csv(Path("reports") / "pems08_communities.csv", index=False)
print("communities saved → reports/pems08_communities.csv")

def fig_centrality_map(values, title, fname, cmap="plasma"):
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.15, arrows=False, width=0.5, ax=ax)
    sc = nx.draw_networkx_nodes(G, pos, node_size=40, node_color=values, cmap=cmap, ax=ax)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    out = REPORTS / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved → {out}")

for col, title, fname, cmap in [
    ("betweenness", "Betweenness centrality", "pems08_betweenness.png", "plasma"),
    ("closeness",   "Closeness centrality",   "pems08_closeness.png",   "viridis"),
    ("eigenvector", "Eigenvector centrality", "pems08_eigenvector.png", "magma"),
    ("pagerank",    "PageRank",               "pems08_pagerank.png",    "inferno"),
]:
    fig_centrality_map(cent[col].values, f"PEMS08 — {title}", fname, cmap=cmap)

cols = ["betweenness", "closeness", "eigenvector", "pagerank"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, cols):
    ax.hist(cent[col], bins=25, edgecolor="black", color="steelblue")
    ax.set_title(col.capitalize())
    ax.set_xlabel("Value")
    ax.set_ylabel("Sensors")
    ax.grid(alpha=0.3)
fig.suptitle("PEMS08 — centrality distributions", fontsize=13)
fig.tight_layout()
out = REPORTS / "pems08_centrality_distributions.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved → {out}")

cmap_comm = matplotlib.colormaps.get_cmap("tab20").resampled(n_comm)
colors    = [cmap_comm(c) for c in comm]
fig, ax   = plt.subplots(figsize=(8, 8))
nx.draw_networkx_edges(G, pos, alpha=0.15, arrows=False, width=0.5, ax=ax)
nx.draw_networkx_nodes(G, pos, node_size=40, node_color=colors, ax=ax)
ax.set_title(f"PEMS08 — Louvain communities (k={n_comm})")
ax.axis("off")
handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap_comm(i), markersize=8, label=f"C{i}")
    for i in range(n_comm)
]
ax.legend(handles=handles, loc="lower left", fontsize=7,
          ncol=max(1, n_comm // 8), framealpha=0.7)
fig.tight_layout()
out = REPORTS / "pems08_communities.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved → {out}")

print("\nRunning robustness simulation …")

def robustness(strategy: str, fraction: float = 0.5, step: int = 3):
    G_work   = G.copy()
    n_nodes  = G.number_of_nodes()
    n_remove = int(n_nodes * fraction)

    if strategy == "betweenness":
        scores = nx.betweenness_centrality(G_work, normalized=True, weight=None)
        order  = sorted(scores, key=scores.__getitem__, reverse=True)
    elif strategy == "degree":
        order  = sorted(G_work.nodes(), key=lambda n: G_work.degree(n), reverse=True)
    else:
        rng   = np.random.default_rng(0)
        order = list(rng.permutation(sorted(G_work.nodes())))

    def _lcc():
        comps = nx.weakly_connected_components(G_work)
        return max((len(c) for c in comps), default=0)

    records = [{"removed_fraction": 0.0, "lcc_fraction": _lcc() / n_nodes}]
    removed = 0
    for node in order[:n_remove]:
        if node in G_work:
            G_work.remove_node(node)
        removed += 1
        if removed % step == 0 or removed == n_remove:
            rem = G_work.number_of_nodes()
            records.append({
                "removed_fraction": removed / n_nodes,
                "lcc_fraction":     _lcc() / rem if rem > 0 else 0.0,
            })
    return pd.DataFrame(records)

strategies = ["betweenness", "degree", "random"]
labels_map = {"betweenness": "Betweenness attack", "degree": "Degree attack", "random": "Random removal"}
colors_map = {"betweenness": "red", "degree": "darkorange", "random": "steelblue"}

fig, ax = plt.subplots(figsize=(8, 5))
for strat in strategies:
    df = robustness(strat)
    ax.plot(df["removed_fraction"], df["lcc_fraction"],
            label=labels_map[strat], color=colors_map[strat], lw=2)
ax.set_xlabel("Fraction of nodes removed")
ax.set_ylabel("LCC size / remaining nodes")
ax.set_title("PEMS08 — robustness under targeted vs. random removal")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out = REPORTS / "pems08_robustness.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved → {out}")

n        = G_un.number_of_nodes()
m        = G_un.number_of_edges()
p        = m / (n * (n - 1) / 2)
real_cc  = nx.average_clustering(G_un)
er_ccs   = [nx.average_clustering(nx.erdos_renyi_graph(n, p, seed=s)) for s in range(5)]
er_mean  = np.mean(er_ccs)
er_std   = np.std(er_ccs)
lcc_size = len(max(nx.connected_components(G_un), key=len))

print("\n--- Null model comparison (undirected) ---")
print(f"Real graph:  n={n}  m={m}  p_eff={p:.5f}")
print(f"Clustering:  real={real_cc:.4f}   ER={er_mean:.4f}±{er_std:.4f}")
print(f"LCC size:    {lcc_size}/{n} = {lcc_size/n:.2%}")

print("\nDone.")
