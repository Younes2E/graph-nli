import matplotlib.pyplot as plt
from utils.graph import extract_triples_propositions, extract_triples
import networkx as nx
import matplotlib
import os
 



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
    "They grew up in San Fransisco and now they both live in New York. One of them has an appartment in Manhattan.",
    "Šafov is a village and municipality (obec) in Znojmo District in the South Moravian Region of the Czech Republic."
]

if __name__ == "__main__":
    for i, p in enumerate(phrases):
        g_p = extract_triples_propositions(p)
        plot_knowledge_graph(g_p, text = phrases[i], file = f"images/graphes_test/propositionned/graph_{i}.png")
        g_q = extract_triples(p)
        plot_knowledge_graph(g_q, text=phrases[i], file = f"images/graphes_test/direct/graph_{i}.png")



    """

    dataset = load_dataset("stanfordnlp/snli")
    snli_test = dataset["test"].filter(lambda x: x["label"] != -1)
    data = snli_test["hypothesis"][:10]
    for i, p in enumerate(data):
        g = build_graph(p)
        plot_knowledge_graph(g,  f"images/graphes_snli/graph_{i}.png", data[i])

    """