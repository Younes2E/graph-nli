# Dans ton nouveau fichier (ex: interface.py)
from graph import build_graph
import networkx as nx
import matplotlib.pyplot as plt




def plot_knowledge_graph(triplets, n, text):
    """
    Prend en entrée un ensemble de triplets (S, R, O) et l'affiche sous forme de graphe orienté.
    """
    if not triplets:
        print("Le graphe est vide, rien à afficher.")
        return

    # 1. Création du graphe orienté NetworkX
    G = nx.DiGraph()
    
    # Dictionnaire pour stocker les labels des arêtes (les relations)
    edge_labels = {}
    
    for subj, rel, obj in triplets:
        G.add_edge(subj, obj)
        edge_labels[(subj, obj)] = rel

    # 2. Calcul du layout (positionnement des nœuds)
    # spring_layout simule des forces physiques pour espacer les nœuds proprement
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1.5, seed=42) 

    # 3. Dessin des composants du graphe
    # Dessin des nœuds
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="skyblue", alpha=0.9)
    
    # Dessin des liens (flèches)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray", min_source_margin=30, min_target_margin=30,width=1.5)
    
    # Dessin des étiquettes des nœuds (Sujets et Objets)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_family="sans-serif")
    
    # Dessin des étiquettes des arêtes (Relations)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="darkred")

    # 4. Affichage / Sauvegarde 
    plt.title(f"{text}", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    
    # AU LIEU DE plt.show() :
    import os
    # On force Matplotlib à ne pas chercher d'interface graphique (évite les crashs sur serveur)
    import matplotlib
    matplotlib.use('Agg') 
    
    nom_fichier = f"images/mon_graphe{n}.png"
    plt.savefig(nom_fichier, format="png", dpi=300)
    plt.close() # Libère la mémoire
    print(f"Graphe sauvegardé avec succès dans : {os.path.abspath(nom_fichier)}")


texts = ["Frank is in New York, but he plays football in Manchester.","The cat and the dog are in the kitchen.", "John is eating a sandwich while Claire eats a pizza","Frank's dog doesn't eat fruits, he is allergic","Simon didn't call me back, he is busy.", "Younes is working on a project. His friend is playing a video game.", "I have never seen anyone like Frank, he must be gifted.","He must be sick.", "He is sick.", "He played football after eating.","A cat and a dog are in the kitchen","Two dogs are eating and playing football"]

for i, p in enumerate(texts):
    mon_graphe = build_graph(p)
    plot_knowledge_graph(mon_graphe, i, texts[i])
