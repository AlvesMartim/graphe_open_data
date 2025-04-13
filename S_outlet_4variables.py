import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # Assurez-vous que vous avez installé python-louvain

# Charger les données
df = pd.read_csv("euvsdisinfo_v1_2.csv", delimiter="\t")

# Limiter le nombre d'outlets à 50 les plus fréquents
top_outlets = df['outlet'].value_counts().head(50).index.tolist()

# Créer un graphe vide avec uniquement les outlets comme sommets
G = nx.Graph()
G.add_nodes_from(top_outlets)

def country_similarity(country1, country2, df):
    # Filtrer les données pour chaque pays
    data1 = df[df['normalized_country'] == country1]
    data2 = df[df['normalized_country'] == country2]

    # Initialiser le score de similarité
    score = 0

    # Langues cibles
    lang1 = set(data1['target_language'].dropna().unique())
    lang2 = set(data2['target_language'].dropna().unique())
    shared_languages = lang1.intersection(lang2)
    lang_score = len(shared_languages)

    # Mots-clés
    keywords1 = set(','.join(data1['keywords'].dropna()).split(','))
    keywords2 = set(','.join(data2['keywords'].dropna()).split(','))
    shared_keywords = keywords1.intersection(keywords2)
    keyword_score = len(shared_keywords)

    # Pays d'origine
    origin1 = set(data1['origin_country'].dropna().unique())
    origin2 = set(data2['origin_country'].dropna().unique())
    shared_origin = origin1.intersection(origin2)
    origin_score = len(shared_origin)

    # Dates (année de publication)
    years1 = set(pd.to_datetime(data1['date_published'], errors='coerce').dropna().dt.year)
    years2 = set(pd.to_datetime(data2['date_published'], errors='coerce').dropna().dt.year)
    shared_years = years1.intersection(years2)
    year_score = len(shared_years)

    # Pondération (ajustable selon la pertinence)
    score += lang_score * 4           # langues cibles = poids fort
    score += keyword_score * 2        # mots-clés = importance moyenne
    score += origin_score * 3         # pays d'origine = poids fort
    score += year_score * 1           # années communes = faible importance

    return score

# Ajouter des arêtes entre les outlets en fonction de la similarité
for i, outlet1 in enumerate(top_outlets):
    for j, outlet2 in enumerate(top_outlets):
        if i < j:  # Éviter de comparer un outlet avec lui-même
            sim = similarity(outlet1, outlet2, df)
            if sim > 0:  # Ajouter une arête seulement si la similarité est supérieure à 0
                # Ajouter une arête avec un poids en fonction de la similarité
                G.add_edge(outlet1, outlet2, weight=sim)

# Supprimer les sommets non connexes (ceux qui ne sont pas dans le plus grand composant connexe)
components = list(nx.connected_components(G))  # Obtenir les composants connexes
largest_component = max(components, key=len)  # Trouver le plus grand composant connexe

# Créer un sous-graphe avec seulement les sommets du plus grand composant connexe
G_subgraph = G.subgraph(largest_component)

# Détection de communautés - Louvain (avec python-louvain)
partition = community_louvain.best_partition(G_subgraph)

# Dessiner le graphe avec les communautés détectées
pos = nx.spring_layout(G_subgraph)
plt.figure(figsize=(12, 12))

# Dessiner les nœuds avec des couleurs différentes selon leur communauté
community_colors = [partition[node] for node in G_subgraph.nodes]
nx.draw_networkx_nodes(G_subgraph, pos, node_color=community_colors, cmap=plt.cm.rainbow, node_size=300)
nx.draw_networkx_edges(G_subgraph, pos, alpha=0.3)
nx.draw_networkx_labels(G_subgraph, pos, font_size=10)

plt.title("Graphe de similarité des Outlets avec Détection de Communautés (Composant Connexe)")
plt.axis('off')
plt.show()
