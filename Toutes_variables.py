import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
from collections import defaultdict
from tqdm import tqdm

# Charger les données
df = pd.read_csv("euvsdisinfo_v1_2.csv", delimiter="\t")

# Limiter aux 50 outlets les plus fréquents
top_outlets = df['outlet'].value_counts().head(600).index.tolist()

# Graphe initial
G = nx.Graph()
G.add_nodes_from(top_outlets)

# Fonction de similarité Jaccard
def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

# Fonction de similarité avancée
def advanced_similarity(outlet1, outlet2, df):
    data1 = df[df['outlet'] == outlet1]
    data2 = df[df['outlet'] == outlet2]

    keywords1 = set(','.join(data1['keywords'].dropna()).split(','))
    keywords2 = set(','.join(data2['keywords'].dropna()).split(','))
    text_sim = jaccard_similarity(keywords1, keywords2)

    geo1 = set(data1['country'].dropna())
    geo2 = set(data2['country'].dropna())
    geo_sim = jaccard_similarity(geo1, geo2)

    lang1 = set(data1['target_language'].dropna())
    lang2 = set(data2['target_language'].dropna())
    lang_sim = jaccard_similarity(lang1, lang2)

    years1 = set(pd.to_datetime(data1['publication_date'], dayfirst=True, errors='coerce').dropna().dt.year)
    years2 = set(pd.to_datetime(data2['publication_date'], dayfirst=True, errors='coerce').dropna().dt.year)
    time_sim = jaccard_similarity(years1, years2)

    total_score = (
        text_sim * 0.4 +
        geo_sim * 0.25 +
        lang_sim * 0.2 +
        time_sim * 0.15
    )
    return total_score

# Construction du graphe avec barre de progression
SIM_THRESHOLD = 0.2
n = len(top_outlets)
total_pairs = (n * (n - 1)) // 2

print("Calcul des similarités et construction du graphe...")

with tqdm(total=total_pairs) as pbar:
    for i, o1 in enumerate(top_outlets):
        for j in range(i + 1, n):
            o2 = top_outlets[j]
            sim = advanced_similarity(o1, o2, df)
            if sim >= SIM_THRESHOLD:
                G.add_edge(o1, o2, weight=sim)
            pbar.update(1)

# Extraction du plus grand composant connexe
components = list(nx.connected_components(G))
if not components:
    print("Aucun composant connexe trouvé.")
    exit()

largest_component = max(components, key=len)
G_subgraph = G.subgraph(largest_component)

# Détection de communautés (Louvain)
partition = community_louvain.best_partition(G_subgraph)

# Préparer les positions
pos = nx.spring_layout(G_subgraph, seed=42)

# Couleurs de communautés
community_colors = [partition[node] for node in G_subgraph.nodes]

# Dessiner les nœuds et les arêtes
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(G_subgraph, pos, node_color=community_colors, cmap=plt.cm.rainbow, node_size=300)
nx.draw_networkx_edges(G_subgraph, pos, alpha=0.3)

# Afficher seulement 10 labels max par communauté
labels_to_display = {}
seen_by_community = defaultdict(int)
for node in G_subgraph.nodes():
    comm = partition[node]
    if seen_by_community[comm] < 25:
        labels_to_display[node] = node
        seen_by_community[comm] += 1

nx.draw_networkx_labels(G_subgraph, pos, labels=labels_to_display, font_size=10)

plt.title("Détection de communautés parmi les Outlets (10 labels max/communauté)")
plt.axis('off')
plt.tight_layout()
plt.show()
