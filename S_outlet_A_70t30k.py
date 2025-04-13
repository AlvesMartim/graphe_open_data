import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # Assurez-vous que vous avez installé python-louvain

# Charger les données
df = pd.read_csv("euvsdisinfo_v1_2.csv", delimiter="\t")

# Limiter le nombre d'outlets à 50 les plus fréquents
top_outlets = df['outlet'].value_counts().head(100).index.tolist()

# Créer un graphe vide avec uniquement les outlets comme sommets
G = nx.Graph()
G.add_nodes_from(top_outlets)

# Définir la fonction de similarité personnalisée
def similarity(outlet1, outlet2, df):
    # Filtrer le DataFrame pour obtenir les données associées à chaque outlet
    data1 = df[df['outlet'] == outlet1].iloc[0]  # On prend la première occurrence
    data2 = df[df['outlet'] == outlet2].iloc[0]  # On prend la première occurrence
    
    # Initialiser le score de similarité
    score = 0
    
    # Vérifier la similarité sur 'target_language'
    languages1 = [language.strip() for language in str(data1['target_language']).split(',')] if pd.notna(data1['target_language']) else []
    languages2 = [language.strip() for language in str(data2['target_language']).split(',')] if pd.notna(data2['target_language']) else []
    
    # Ajouter 7 points si les langues cibles sont communes
    if any(language in languages1 for language in languages2):
        score += 7

    # Vérifier la similarité sur 'keyword'
    keywords1 = [keyword.strip() for keyword in str(data1['keywords']).split(',')] if pd.notna(data1['keywords']) else []
    keywords2 = [keyword.strip() for keyword in str(data2['keywords']).split(',')] if pd.notna(data2['keywords']) else []
    
    # Ajouter 3 points si les mots-clés sont communs
    if any(keyword in keywords1 for keyword in keywords2):
        score += 3

    # Retourner le score de similarité
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
nx.draw_networkx_nodes(G_subgraph, pos, node_color=community_colors, cmap=plt.cm.rainbow, node_size=50)
nx.draw_networkx_edges(G_subgraph, pos, alpha=0.15)
nx.draw_networkx_labels(G_subgraph, pos, font_size=7)

plt.title("Graphe de similarité des Outlets (Urls) avec Détection de Communautés")
plt.axis('off')
plt.show()
