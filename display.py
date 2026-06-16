from datasets import load_dataset
import matplotlib.pyplot as plt
from graph import build_graph, extract_triplets_qwen
import networkx as nx
import matplotlib
import os


def plot_knowledge_graph(triplets, nom_fichier, text):
    if not triplets:
        print("Graph vide")
        return

    G = nx.DiGraph()
    edge_labels = {}
    
    for subj, rel, obj in triplets:
        G.add_edge(subj, obj)
        edge_labels[(subj, obj)] = rel

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1.5, seed=42) 

    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="skyblue", alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray", min_source_margin=30, min_target_margin=30,width=1.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_family="sans-serif")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="darkred")

    plt.title(f"{text}", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    
    matplotlib.use('Agg') 
    
    
    plt.savefig(nom_fichier, format="png", dpi=300)
    plt.close() 
    print(f"Graphe sauvegardé avec succès dans : {os.path.abspath(nom_fichier)}")



phrases = [
    "Luc is working on a project with his interns, Younes and Ava. They all work at LISN.",
    "Steve Jobs founded Apple.",
    "He doesn't play baseball.", 
    "Chris doesn't like staying with us.",
    "Two boys are playing outside", 
    "A man is playing an instrument.", 
    "He and his friend played at the park, they played football and basketball.",
    "The cat and the dog are in the kitchen.", 
    "Frank's dog doesn't eat fruits, he is allergic",
    "Simon didn't call me back, he is busy.", 
    "Younes is working on a project. His friend is playing a video game.", 
    "I have never seen anyone like Frank, he must be gifted.",
    "He must be sick.", 
    "He is sick.", 
    "He played football after eating.",
    "Two dogs are eating and playing football",
    "Frank is in New York, but he plays football in Manchester.",
    "John is eating a sandwich while Claire eats a pizza",
    "John is eating a ham pizza and a chicken sandwich.",
    "John is eating meat.",
    "They grew up in San Fransisco and now they both live in New York. One of them has an appartment in Manhattan."
]

if __name__ == "__main__":
    for i, p in enumerate(phrases):
        g_p = build_graph(p)
        plot_knowledge_graph(g_p,  f"images/graphes_test/propositionned/graph_{i}.png", phrases[i])
        g_q = extract_triplets_qwen(p)
        plot_knowledge_graph(g_q,  f"images/graphes_test/direct/graph_{i}.png", phrases[i])



    """

    dataset = load_dataset("stanfordnlp/snli")
    snli_test = dataset["test"].filter(lambda x: x["label"] != -1)
    data = snli_test["hypothesis"][:10]
    for i, p in enumerate(data):
        g = build_graph(p)
        plot_knowledge_graph(g,  f"images/graphes_snli/graph_{i}.png", data[i])

    """