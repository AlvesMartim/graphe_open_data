import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # Assurez-vous que vous avez installé python-louvain

# Charger les données
df = pd.read_csv("euvsdisinfo_v1_2.csv", delimiter="\t")

# Limiter le nombre de mots-clés à 50 les plus fréquents
top_keywords = df['keywords'].value_counts().head(50).index.tolist()

# Créer un graphe vide avec uniquement les mots-clés comme sommets
G = nx.Graph()
G.add_nodes_from(top_keywords)

# Définir la fonction de similarité personnalisée
def similarity(keyword1, keyword2, df):
    # Filtrer le DataFrame pour obtenir les pays et les langues cibles associés à chaque mot-clé
    data1 = df[df['keywords'] == keyword1].iloc[0]  # On prend la première occurrence
    data2 = df[df['keywords'] == keyword2].iloc[0]  # On prend la première occurrence
    
    # Initialiser le score de similarité
    score = 0
    
    # Vérifier si 'country' n'est pas NaN et diviser les pays par une virgule
    countries1 = [country.strip() for country in str(data1['country']).split(',')] if pd.notna(data1['country']) else []
    countries2 = [country.strip() for country in str(data2['country']).split(',')] if pd.notna(data2['country']) else []
    
    # Comparer les pays : on vérifie s'il y a des pays communs
    if any(country in countries1 for country in countries2):
        score += 1  # Ajouter 1 si au moins un pays est commun

    # Vérifier si 'target_language' n'est pas NaN et diviser les langues cibles par une virgule
    languages1 = [language.strip() for language in str(data1['target_language']).split(',')] if pd.notna(data1['target_language']) else []
    languages2 = [language.strip() for language in str(data2['target_language']).split(',')] if pd.notna(data2['target_language']) else []
    
    # Comparer les langues cibles : on vérifie s'il y a des langues cibles communes
    if any(language in languages1 for language in languages2):
        score += 1  # Ajouter 1 si au moins une langue cible est commune

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

plt.title("Graphe de similarité des Keywords avec Détection de Communautés (Composant Connexe)")
plt.axis('off')
plt.show()
