import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain

# Charger les données
df = pd.read_csv("euvsdisinfo_v1_2.csv", delimiter="\t")

# Fonction pour nettoyer et normaliser les noms de pays
def normalize_country_name(country_str):
    # Séparer les pays s'ils sont combinés
    countries = [c.strip() for c in str(country_str).split(',')]
    # Retourner le premier pays (ou le pays unique)
    return countries[0]

# Normaliser les noms de pays
df['normalized_country'] = df['country'].apply(normalize_country_name)

# Sélectionner les 100 pays les plus fréquents
top_countries = df['normalized_country'].value_counts().head(100).index.tolist()

# Filtrer le dataframe pour ne garder que ces pays
df_filtered = df[df['normalized_country'].isin(top_countries)]

# Créer un graphe vide avec uniquement les pays top 100 comme sommets
G = nx.Graph()
G.add_nodes_from(top_countries)

# Fonction de similarité personnalisée pour les pays
def country_similarity(country1, country2, df):
    # Filtrer les données pour chaque pays
    data1 = df[df['normalized_country'] == country1]
    data2 = df[df['normalized_country'] == country2]
    
    # Initialiser le score de similarité
    score = 0
    
    # Vérifier la similarité des langues cibles
    languages1 = set(data1['target_language'].unique())
    languages2 = set(data2['target_language'].unique())
    score += len(languages1.intersection(languages2)) * 3
    
    # Vérifier la similarité des mots-clés
    keywords1 = set(','.join(data1['keywords'].dropna()).split(','))
    keywords2 = set(','.join(data2['keywords'].dropna()).split(','))
    score += len(keywords1.intersection(keywords2)) * 2
    
    return score

# Ajouter des arêtes entre les pays en fonction de la similarité
for i, country1 in enumerate(top_countries):
    for j, country2 in enumerate(top_countries):
        if i < j:  # Éviter de comparer un pays avec lui-même
            sim = country_similarity(country1, country2, df_filtered)
            if sim > 0:  # Ajouter une arête seulement si la similarité est supérieure à 0
                G.add_edge(country1, country2, weight=sim)

# Supprimer les sommets non connexes
components = list(nx.connected_components(G))
largest_component = max(components, key=len)
G_subgraph = G.subgraph(largest_component)

# Détection de communautés - Algorithme de Louvain
partition = community_louvain.best_partition(G_subgraph)

# Visualisation du graphe
plt.figure(figsize=(20, 20))
pos = nx.spring_layout(G_subgraph, k=0.5)  # k contrôle l'espacement

# Dessiner les nœuds avec des couleurs différentes selon leur communauté
community_colors = [partition[node] for node in G_subgraph.nodes()]
nx.draw_networkx_nodes(G_subgraph, pos, node_color=community_colors, cmap=plt.cm.rainbow, node_size=300)
nx.draw_networkx_edges(G_subgraph, pos, edge_color='gray', alpha=0.1, width=1)
nx.draw_networkx_labels(G_subgraph, pos, font_size=8, font_weight="bold")

plt.title("Graphe de Similarité des 100 Pays les Plus Fréquents dans la Désinformation", fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()

# Analyse des communautés
print("Communautés détectées :")
for community_id in set(partition.values()):
    members = [country for country, com_id in partition.items() if com_id == community_id]
    print(f"Communauté {community_id}: {members}")



# Afficher le nombre de pays dans chaque communauté
community_sizes = {}
for community_id in set(partition.values()):
    community_sizes[community_id] = sum(1 for country, com_id in partition.items() if com_id == community_id)
print("\nTaille des communautés :")
for community_id, size in sorted(community_sizes.items()):
    print(f"Communauté {community_id}: {size} pays")