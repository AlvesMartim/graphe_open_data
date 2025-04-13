import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# Charger les données
df = pd.read_csv("euvsdisinfo_v1_2.csv", delimiter="\t")

# Paramètres ajustables
TOP_OUTLETS = 3000
SIM_THRESHOLD = 0.25
MAX_LABELS_PER_COMM = 5

# Limiter aux outlets les plus fréquents
top_outlets = df['outlet'].value_counts().head(TOP_OUTLETS).index.tolist()

# Graphe initial
G = nx.Graph()
G.add_nodes_from(top_outlets)

# Fonction de similarité optimisée
def advanced_similarity(outlet1, outlet2, df):
    data1 = df[df['outlet'] == outlet1]
    data2 = df[df['outlet'] == outlet2]

    keywords1 = set(','.join(data1['keywords'].fillna('')).split(','))
    keywords2 = set(','.join(data2['keywords'].fillna('')).split(','))
    text_sim = len(keywords1 & keywords2) / len(keywords1 | keywords2) if (keywords1 and keywords2) else 0

    geo_sim = len(set(data1['country']) & set(data2['country'])) / len(set(data1['country']) | set(data2['country'])) if (not data1['country'].empty and not data2['country'].empty) else 0

    return 0.6 * text_sim + 0.4 * geo_sim

# Construction du graphe
print("Construction du graphe...")
for i, o1 in enumerate(tqdm(top_outlets)):
    for j in range(i + 1, len(top_outlets)):
        o2 = top_outlets[j]
        sim = advanced_similarity(o1, o2, df)
        if sim >= SIM_THRESHOLD:
            G.add_edge(o1, o2, weight=sim)

# Extraction du composant principal
largest_component = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_component)

# Détection de communautés
partition = community_louvain.best_partition(G, resolution=0.9)

# Création de la figure avec espace pour colorbar
plt.figure(figsize=(16, 12))
plt.rcParams.update({'font.size': 8})
grid = plt.GridSpec(1, 2, width_ratios=[0.9, 0.05])
ax = plt.subplot(grid[0])

# Layout
pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)

# Couleurs et tailles
cmap = plt.cm.tab20
node_colors = [partition[n] for n in G.nodes()]
degrees = dict(G.degree())
node_sizes = [300 + 1000 * degrees[n]/max(degrees.values()) for n in G.nodes()]

# Dessin du graphe
nx.draw_networkx_nodes(G, pos,
                      node_color=node_colors,
                      cmap=cmap,
                      node_size=node_sizes,
                      alpha=0.9,
                      edgecolors='black',
                      linewidths=0.5,
                      ax=ax)

edge_weights = [G[u][v]['weight']*2 for u,v in G.edges()]
nx.draw_networkx_edges(G, pos,
                      width=edge_weights,
                      alpha=0.1 + 0.3*np.array(edge_weights)/max(edge_weights),
                      edge_color='lightgray',
                      ax=ax)

# Labels stratégiques
labels_to_show = {}
comm_counts = defaultdict(int)
central_nodes = sorted(G.nodes(), key=lambda x: degrees[x], reverse=True)

for node in central_nodes:
    comm = partition[node]
    if comm_counts[comm] < MAX_LABELS_PER_COMM:
        labels_to_show[node] = node.split('.')[0][:15]  # Raccourcir davantage
        comm_counts[comm] += 1

nx.draw_networkx_labels(G, pos,
                       labels=labels_to_show,
                       font_size=9,
                       font_weight='bold',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                       ax=ax)

# Colorbar
cax = plt.subplot(grid[1])
sm = plt.cm.ScalarMappable(cmap=cmap,
                          norm=plt.Normalize(vmin=min(node_colors),
                                           vmax=max(node_colors)))
sm.set_array([])
plt.colorbar(sm, cax=cax, label='Communauté')

# Titre et ajustement
plt.suptitle(f"Réseau des {len(G.nodes())} principaux outlets\n"
            f"Communautés détectées (seuil: {SIM_THRESHOLD})",
            y=0.95, fontsize=12)
plt.tight_layout()
plt.savefig('network_final.png', dpi=300, bbox_inches='tight')
plt.show()