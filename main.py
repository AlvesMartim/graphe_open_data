import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain 

# Charger les données
df = pd.read_csv("euvsdisinfo_v1_2.csv", delimiter="\t")

# Limiter le nombre de mots-clés à 50 les plus fréquents
top_keywords = df['keywords'].value_counts().head(50).index.tolist()

# Créer un graphe vide avec uniquement les mots-clés comme sommets
G = nx.Graph()
G.add_nodes_from(top_keywords)

# Définir la fonction de similarité personnalisée
# Nous allons comparer les mots-clés selon les attributs 'country' et 'target_language'

def similarity(keyword1, keyword2, df):
    # Filtrer le DataFrame pour obtenir les pays et les langues cibles associés à chaque mot-clé
    data1 = df[df['keywords'] == keyword1].iloc[0]  # On prend la première occurrence
    data2 = df[df['keywords'] == keyword2].iloc[0]  # On prend la première occurrence
    
    # Initialiser le score de similarité
    score = 0
    
    # Comparer les attributs 'country' et 'target_language'
    if data1['country'] == data2['country']:
        score += 1  # Ajouter 1 si les pays sont les mêmes
    if data1['target_language'] == data2['target_language']:
        score += 1  # Ajouter 1 si les langues cibles sont les mêmes

    # Retourner le score de similarité
    return score

# Ajouter des arêtes entre les mots-clés en fonction de la similarité
for i, keyword1 in enumerate(top_keywords):
    for j, keyword2 in enumerate(top_keywords):
        if i < j:  # Éviter de comparer un mot-clé avec lui-même
            sim = similarity(keyword1, keyword2, df)
            if sim > 0:  # Ajouter une arête seulement si la similarité est supérieure à 0
                # Ajouter une arête avec un poids en fonction de la similarité
                G.add_edge(keyword1, keyword2, weight=sim)

# Détection de communautés - Louvain (avec python-louvain)
partition = community_louvain.best_partition(G)

# Dessiner le graphe avec les communautés détectées
pos = nx.spring_layout(G)
plt.figure(figsize=(12, 12))

# Dessiner les nœuds avec des couleurs différentes selon leur communauté
community_colors = [partition[node] for node in G.nodes]
nx.draw_networkx_nodes(G, pos, node_color=community_colors, cmap=plt.cm.rainbow, node_size=300)
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_labels(G, pos, font_size=10)

plt.title("Graphe de similarité des Keywords avec Détection de Communautés")
plt.axis('off')
plt.show()
