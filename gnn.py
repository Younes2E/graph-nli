import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv, global_max_pool, global_mean_pool
import torch
import pandas as pd
import numpy as np
import json

LANG = 'en'

with open("data/relations.json") as f:
    rel_json = json.load(f)

rel_conceptnet = rel_json[f"concept_{LANG}"]
rel_closedie = rel_json[f"closed_ie"].keys()

rel = set(rel_closedie) | set(rel_conceptnet)


class RGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations, num_heads=4, out_classes=3):
        super(RGAT, self).__init__()
        
        # 1. Couches de Message Passing (RGAT)
        # in_channels : taille de l'embedding initial d'un nœud
        # num_relations : len(rel) de ton script
        self.rgat1 = RGATConv(in_channels, hidden_channels, num_relations, heads=num_heads, concat=False)
        self.rgat2 = RGATConv(hidden_channels, hidden_channels, num_relations, heads=num_heads, concat=False)
        
        # 2. Tête de classification NLI (Prémisse vs Hypothèse)
        # Entrée = Concat(G_P, G_H, |G_P - G_H|, G_P * G_H) -> Standard NLI heuristic
        classifier_input_dim = hidden_channels * 4
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, out_classes) # 3 classes: Entailment, Neutral, Contradiction
        )

    def forward_graph(self, x, edge_index, edge_type, batch):
        # x : Features des nœuds [num_nodes, in_channels]
        # edge_index : Topologie du graphe [2, num_edges]
        # edge_type : Type de chaque arête [num_edges] (entiers mappés depuis ton set `rel`)
        # batch : Vecteur d'assignation des nœuds aux graphes du batch
        
        # Message Passing
        x = self.rgat1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.rgat2(x, edge_index, edge_type)
        x = F.relu(x)
        
        # Pooling : Réduction du graphe variable en un vecteur fixe
        # Une combinaison de Mean et Max pooling capture souvent mieux la sémantique
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        return x_mean + x_max # Vecteur global du graphe [batch_size, hidden_channels]

    def forward(self, premise_data, hypothesis_data):
        # Encodage du graphe de la Prémisse
        g_p = self.forward_graph(
            premise_data.x, 
            premise_data.edge_index, 
            premise_data.edge_type, 
            premise_data.batch
        )
        
        # Encodage du graphe de l'Hypothèse
        g_h = self.forward_graph(
            hypothesis_data.x, 
            hypothesis_data.edge_index, 
            hypothesis_data.edge_type, 
            hypothesis_data.batch
        )
        
        # Heuristique NLI standard pour la comparaison
        diff = torch.abs(g_p - g_h)
        prod = g_p * g_h
        features = torch.cat([g_p, g_h, diff, prod], dim=1)
        
        # Classification
        logits = self.classifier(features)
        return logits

