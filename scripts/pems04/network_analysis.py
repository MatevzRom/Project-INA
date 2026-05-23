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

data = np.load("data/PEMS04/PEMS04.npz")["data"].astype(np.float32)
edges = pd.read_csv("data/PEMS04/PEMS04.csv").rename(columns={"cost": "distance"})
T, N, _ = data.shape

G = nx.DiGraph()
for n in range(N):
    G.add_node(n)
distances = edges["distance"].to_numpy()
sigma = float(distances.std())
for _, row in edges.iterrows():
    d = float(row["distance"])
    w = float(np.exp(-(d ** 2) / (sigma ** 2)))
    G.add_edge(int(row["from"]), int(row["to"]), distance=d, weight=w)

nodes = sorted(G.nodes())
G_un = G.to_undirected()

print("Computing spring layout …")
pos = nx.spring_layout(G_un, seed=42, iterations=300)

print("Computing centralities …")
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())
bet = nx.betweenness_centrality(G, normalized=True, weight=None)
clo = nx.closeness_centrality(G)
lwcc = max(nx.connected_components(G_un), key=len)
eig_lwcc = nx.eigenvector_centrality_numpy(G_un.subgraph(lwcc), weight=None)
eig = {n: eig_lwcc.get(n, 0.0) for n in G.nodes()}
pr = nx.pagerank(G, alpha=0.85, weight=None)

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
cent.to_csv(Path("reports") / "pems04_centralities.csv")
print(f"centralities saved → reports/pems04_centralities.csv")

print("Detecting communities …")
partition = community_louvain.best_partition(G_un, random_state=42)
comm = np.array([partition[n] for n in nodes], dtype=np.int32)
n_comm = comm.max() + 1
print(f"Louvain: {n_comm} communities")

print("\n--- Centrality summary ---")
print(cent.describe().to_string())

print("\nTop-10 nodes by betweenness centrality:")
print(cent["betweenness"].sort_values(ascending=False).head(10).to_string())

for col, title, fname, cmap in [
    ("betweenness", "Betweenness centrality", "pems04_betweenness.png", "plasma"),
    ("closeness",   "Closeness centrality",   "pems04_closeness.png",   "viridis"),
    ("eigenvector", "Eigenvector centrality", "pems04_eigenvector.png", "magma"),
    ("pagerank",    "PageRank",               "pems04_pagerank.png",    "inferno"),
]:
    values = cent[col].values
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.15, arrows=False, width=0.5, ax=ax)
    sc = nx.draw_networkx_nodes(G, pos, node_size=30, node_color=values, cmap=cmap, ax=ax)
    plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(f"PEMS04 — {title}")
    ax.axis("off")
    fig.tight_layout()
    out = REPORTS / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved → {out}")

cols = ["betweenness", "closeness", "eigenvector", "pagerank"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, cols):
    ax.hist(cent[col], bins=25, edgecolor="black", color="steelblue")
    ax.set_title(col.capitalize())
    ax.set_xlabel("Value")
    ax.set_ylabel("Sensors")
    ax.grid(alpha=0.3)
fig.suptitle("PEMS04 — centrality distributions", fontsize=13)
fig.tight_layout()
out = REPORTS / "pems04_centrality_distributions.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved → {out}")

cmap = matplotlib.colormaps.get_cmap("tab20").resampled(n_comm)
colors = [cmap(c) for c in comm]
fig, ax = plt.subplots(figsize=(8, 8))
nx.draw_networkx_edges(G, pos, alpha=0.15, arrows=False, width=0.5, ax=ax)
nx.draw_networkx_nodes(G, pos, node_size=30, node_color=colors, ax=ax)
ax.set_title(f"Louvain communities (k={n_comm})")
ax.axis("off")
handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=cmap(i), markersize=8, label=f"C{i}")
    for i in range(n_comm)
]
ax.legend(handles=handles, loc="lower left", fontsize=7,
          ncol=max(1, n_comm // 8), framealpha=0.7)
fig.tight_layout()
out = REPORTS / "pems04_communities.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved → {out}")

strategies = ["betweenness", "degree", "random"]
labels = {"betweenness": "Betweenness attack", "degree": "Degree attack", "random": "Random removal"}
colors_map = {"betweenness": "red", "degree": "darkorange", "random": "steelblue"}

fig, ax = plt.subplots(figsize=(8, 5))
for strat in strategies:
    G_work = G.copy()
    n_remove = int(N * 0.5)
    step = 3
    if strat == "betweenness":
        scores = nx.betweenness_centrality(G_work, normalized=True, weight=None)
        order = sorted(scores, key=scores.__getitem__, reverse=True)
    elif strat == "degree":
        order = sorted(G_work.nodes(), key=lambda n: G_work.degree(n), reverse=True)
    else:
        rng = np.random.default_rng(0)
        order = list(rng.permutation(sorted(G_work.nodes())))

    records = []
    comps = nx.weakly_connected_components(G_work)
    lcc = max((len(c) for c in comps), default=0)
    records.append({"removed_fraction": 0.0, "lcc_fraction": lcc / N})

    removed = 0
    for node in order[:n_remove]:
        if node in G_work:
            G_work.remove_node(node)
        removed += 1
        if removed % step == 0 or removed == n_remove:
            comps = nx.weakly_connected_components(G_work)
            lcc = max((len(c) for c in comps), default=0)
            n_left = G_work.number_of_nodes()
            records.append({
                "removed_fraction": removed / N,
                "lcc_fraction": lcc / n_left if n_left > 0 else 0.0,
            })

    df = pd.DataFrame(records)
    ax.plot(df["removed_fraction"], df["lcc_fraction"],
            label=labels[strat], color=colors_map[strat], lw=2)

ax.set_xlabel("Fraction of nodes removed")
ax.set_ylabel("LCC size / remaining nodes")
ax.set_title("PEMS04 — robustness under targeted vs. random removal")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out = REPORTS / "pems04_robustness.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"saved → {out}")

n_un = G_un.number_of_nodes()
m_un = G_un.number_of_edges()
p = m_un / (n_un * (n_un - 1) / 2)
real_cc = nx.average_clustering(G_un)
er_ccs = []
for seed in range(5):
    er = nx.erdos_renyi_graph(n_un, p, seed=seed)
    er_ccs.append(nx.average_clustering(er))
er_cc_mean = float(np.mean(er_ccs))
er_cc_std = float(np.std(er_ccs))
lcc_nodes = max(nx.connected_components(G_un), key=len)
lcc_frac = len(lcc_nodes) / n_un

print("\n--- Null model comparison (undirected) ---")
print(f"Real graph:  n={n_un}  m={m_un}  p_eff={p:.5f}")
print(f"Clustering:  real={real_cc:.4f}   ER={er_cc_mean:.4f}±{er_cc_std:.4f}")
print(f"LCC size:    {len(lcc_nodes)}/{n_un} = {lcc_frac:.2%}")

comm_df = pd.DataFrame({"node": nodes, "community": comm})
comm_df.to_csv(Path("reports") / "pems04_communities.csv", index=False)
print(f"communities saved → reports/pems04_communities.csv")

print("\nDone.")
