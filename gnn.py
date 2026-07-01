"""Premier test RGAT pour NLI sur les graphes de triplets.

Entraîne un RGAT pour le NLI. Dataset choisi par la constante DATASET ("snli" ou "anli_r1").
Évalue accuracy / F1 / matrice de confusion.

Design :
- features de nœuds et de relations initialisées par all-MiniLM-L6-v2 (384d, gelé),
  projetées par une couche linéaire apprenable -> hidden ;
- un GLOBAL NODE par graphe (init = embedding MiniLM de la phrase entière), relié à tous
  les nœuds via une relation dédiée 'GlobalLink' (=> aucun graphe n'est jamais vide) ;
- RGATConv avec edge_dim (embedding MiniLM de la relation) + num_relations (types) ;
- MATCHING croisé P<->H (Graph Matching Network) : on garde les embeddings de nœuds, chaque
  nœud attend sur l'autre graphe, et le résidu non-matché (≈0 si H⊆P) nourrit le classifieur.

Lancement :  CUDA_VISIBLE_DEVICES=0 python3 gnn.py
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGATConv, global_mean_pool
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

_HERE = os.path.dirname(os.path.abspath(__file__))

# Choix du dataset : "snli" (val->test) ou "anli_r1" (train_r1->test_r1). Un seul mot à changer.
DATASET = "snli"
_DATASETS = {
    "snli":    ("snli_val",      "snli_test"),    # train sur validation, test sur test
    "anli_r1": ("anli_train_r1", "anli_test_r1"),
}
_train_dir, _test_dir = _DATASETS[DATASET]
TRAIN_PATH = os.path.join(_HERE, "out", _train_dir, "graphs.jsonl")
TEST_PATH = os.path.join(_HERE, "out", _test_dir, "graphs.jsonl")

GLOBAL_REL = "GlobalLink"          # relation dédiée global_node <-> nœuds
EMB_DIM = 384                      # dimension all-MiniLM-L6-v2
SEED = 42


# --------------------------------------------------------------------------- #
# Encodeur de features (gelé, caché) — partagé par le dataset.
#   mode="minilm" : embeddings sémantiques all-MiniLM-L6-v2 ;
#   mode="random" : vecteur ALÉATOIRE FIXE par texte (même mot -> même vecteur).
#     -> supprime la sémantique des mots ; il ne reste que la STRUCTURE du graphe.
#        Sert d'ablation : combien le graphe apporte sans le sens des mots.
# --------------------------------------------------------------------------- #
class FeatureEncoder:
    def __init__(self, mode="minilm", device="cpu"):
        self.mode = mode
        self.cache = {}
        if mode == "minilm":
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                             device=device)
        elif mode != "random":
            raise ValueError("mode must be 'minilm' or 'random'.")

    def encode(self, texts):
        """list[str] -> np.ndarray[len, EMB_DIM], avec cache (les mots reviennent souvent)."""
        missing = [t for t in texts if t not in self.cache]
        if missing:
            if self.mode == "minilm":
                vecs = self.model.encode(missing, batch_size=256, show_progress_bar=False,
                                         normalize_embeddings=True)
            else:  # random : vecteur déterministe par texte (seed = hash du texte)
                vecs = []
                for t in missing:
                    rng = np.random.default_rng(abs(hash(t)) % (2**32))
                    v = rng.standard_normal(EMB_DIM).astype(np.float32)
                    vecs.append(v / np.linalg.norm(v))
            for t, v in zip(missing, vecs):
                self.cache[t] = v
        return np.stack([self.cache[t] for t in texts])


# --------------------------------------------------------------------------- #
# Dataset : graphs.jsonl -> paires de graphes PyG (PairData)
# --------------------------------------------------------------------------- #
class PairData(Data):
    """Deux graphes (premise / hypothesis) dans un même objet, préfixes p_ / h_.

    On surcharge __inc__ pour que le batching décale correctement les edge_index de
    chaque sous-graphe selon SON nombre de nœuds (pattern PyG standard pour 2 graphes).
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == "p_edge_index":
            return self.p_x.size(0)
        if key == "h_edge_index":
            return self.h_x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class NLIGraphDataset(Dataset):
    def __init__(self, rows, rel2id, encoder):
        super().__init__()
        self.rows = rows
        self.rel2id = rel2id
        self.enc = encoder

    def len(self):
        return len(self.rows)

    def _build_graph(self, triples, sentence):
        """triples: list[[s,rel,o]] (déjà filtrés sur ie) + sentence -> (x, edge_index, edge_type, edge_attr).

        Nœud 0 = GLOBAL NODE (embedding de la phrase), relié à tous les autres via GlobalLink.
        """
        # entités uniques -> indices (1.. ; 0 réservé au global node)
        ents = []
        idx = {}
        for s, _, o in triples:
            for e in (s, o):
                if e not in idx:
                    idx[e] = len(ents) + 1
                    ents.append(e)

        # features de nœuds : [phrase] + [entités]
        node_texts = [sentence] + ents
        x = torch.tensor(self.enc.encode(node_texts), dtype=torch.float)

        src, dst, etype, e_rel_text = [], [], [], []
        # arêtes des triplets (dirigées s -> o)
        for s, rel, o in triples:
            src.append(idx[s]); dst.append(idx[o])
            etype.append(self.rel2id[rel]); e_rel_text.append(rel)
        # arêtes GlobalLink : global node (0) <-> chaque nœud (bidirectionnel)
        gid = self.rel2id[GLOBAL_REL]
        for n in range(1, len(node_texts)):
            src += [0, n]; dst += [n, 0]
            etype += [gid, gid]; e_rel_text += [GLOBAL_REL, GLOBAL_REL]

        # Cas dégénéré : aucune entité -> graphe réduit au seul global node, sans arête.
        if not e_rel_text:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
            edge_attr = torch.empty((0, EMB_DIM), dtype=torch.float)
        else:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_type = torch.tensor(etype, dtype=torch.long)
            edge_attr = torch.tensor(self.enc.encode(e_rel_text), dtype=torch.float)
        return x, edge_index, edge_type, edge_attr

    def get(self, i):
        r = self.rows[i]
        px, pei, pet, pea = self._build_graph(r["premise_triples"], r["premise"])
        hx, hei, het, hea = self._build_graph(r["hypothesis_triples"], r["hypothesis"])
        # embedding de phrase brut (nœud 0 = global node = MiniLM(sentence)) pour la FUSION
        # texte+graphe ; shape [1, EMB_DIM] -> après batching PyG : [num_graphs, EMB_DIM].
        return PairData(
            p_x=px, p_edge_index=pei, p_edge_type=pet, p_edge_attr=pea,
            h_x=hx, h_edge_index=hei, h_edge_type=het, h_edge_attr=hea,
            p_s=px[0:1], h_s=hx[0:1],
            y=torch.tensor([r["label"]], dtype=torch.long),
        )


# --------------------------------------------------------------------------- #
# Modèle RGAT
# --------------------------------------------------------------------------- #
class RGAT(nn.Module):
    """RGAT + matching croisé P<->H (style Graph Matching Network) + FUSION texte.

    Au lieu de pooler chaque graphe puis comparer les vecteurs (le pooling cassait la
    structure), on garde les embeddings de NŒUDS, puis chaque nœud d'un graphe fait une
    attention croisée sur les nœuds de l'autre. Le RÉSIDU non-matché (`x - x_match`) encode
    directement l'alignement structurel : si H ⊆ P, chaque nœud de H matche bien dans P ->
    résidu H ≈ 0 (signal d'entailment). C'est ce résidu (poolé) qui nourrit le classifieur.

    FUSION texte+graphe (inspiré MHGRN) : le graphe de triplets est incomplet (temps, modalité,
    disjonction, portée de négation...). On CONCATÈNE donc l'embedding MiniLM des phrases P/H
    (s_P, s_H) aux features de matching avant le classifieur : le texte porte ce que le graphe
    rate, le graphe porte le raisonnement structurel.
    """
    def __init__(self, num_relations, hidden=128, heads=4, out_classes=3, dropout=0.2):
        super().__init__()
        self.proj_node = nn.Linear(EMB_DIM, hidden)   # 384 -> hidden (nœuds)
        self.proj_edge = nn.Linear(EMB_DIM, hidden)   # 384 -> hidden (relations)
        self.proj_text = nn.Linear(EMB_DIM, hidden)   # 384 -> hidden (phrases P/H, fusion)
        self.rgat1 = RGATConv(hidden, hidden, num_relations, heads=heads,
                              concat=False, edge_dim=hidden)
        self.rgat2 = RGATConv(hidden, hidden, num_relations, heads=heads,
                              concat=False, edge_dim=hidden)
        self.scale = hidden ** 0.5
        # entrée = matching [pool(P), pool(H), pool(résidu P), pool(résidu H)] (4*hidden)
        #        + texte [s_P, s_H, |s_P-s_H|, s_P*s_H] (4*hidden) = 8*hidden
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 8, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_classes),
        )

    def encode_nodes(self, x, edge_index, edge_type, edge_attr):
        # embeddings de NŒUDS (pas de pooling : on garde la structure)
        x = self.proj_node(x)
        ea = self.proj_edge(edge_attr)
        x = F.relu(self.rgat1(x, edge_index, edge_type, edge_attr=ea))
        x = F.relu(self.rgat2(x, edge_index, edge_type, edge_attr=ea))
        return x

    def _cross_match(self, p_emb, p_batch, h_emb, h_batch, num_graphs):
        """Attention croisée P<->H PAR exemple du batch.
        Renvoie [num_graphs, 4*hidden] = [pool(P), pool(H), pool(résidu P), pool(résidu H)]."""
        d = p_emb.size(1)
        feats = []
        for b in range(num_graphs):
            pb = p_emb[p_batch == b]      # [n_p, d]
            hb = h_emb[h_batch == b]      # [n_h, d]
            if pb.numel() == 0 or hb.numel() == 0:
                feats.append(torch.zeros(4 * d, device=p_emb.device))
                continue
            # chaque nœud de P attend sur les nœuds de H (et inversement)
            a_ph = torch.softmax(pb @ hb.t() / self.scale, dim=1)   # [n_p, n_h]
            a_hp = torch.softmax(hb @ pb.t() / self.scale, dim=1)   # [n_h, n_p]
            p_match = a_ph @ hb            # ce que chaque nœud de P matche dans H
            h_match = a_hp @ pb            # ce que chaque nœud de H matche dans P
            p_res = pb - p_match           # résidu non-matché (structure non alignée)
            h_res = hb - h_match           # ≈ 0 si H ⊆ P
            feats.append(torch.cat([pb.mean(0), hb.mean(0), p_res.mean(0), h_res.mean(0)]))
        return torch.stack(feats)

    def forward(self, data):
        p_emb = self.encode_nodes(data.p_x, data.p_edge_index, data.p_edge_type, data.p_edge_attr)
        h_emb = self.encode_nodes(data.h_x, data.h_edge_index, data.h_edge_type, data.h_edge_attr)
        feats = self._cross_match(p_emb, data.p_x_batch, h_emb, data.h_x_batch, data.num_graphs)
        # FUSION texte : embeddings de phrase P/H projetés -> [s_P, s_H, |s_P-s_H|, s_P*s_H]
        s_p = self.proj_text(data.p_s)          # [num_graphs, hidden]
        s_h = self.proj_text(data.h_s)          # [num_graphs, hidden]
        text_feats = torch.cat([s_p, s_h, (s_p - s_h).abs(), s_p * s_h], dim=1)
        return self.classifier(torch.cat([feats, text_feats], dim=1))


# --------------------------------------------------------------------------- #
# Données : chargement + filtre ie + split
# --------------------------------------------------------------------------- #
def load_rows(path, ie):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        # filtre les triplets dont la relation n'est pas dans ie (données pré-corrections)
        r["premise_triples"] = [t for t in r["premise_triples"] if t[1] in ie]
        r["hypothesis_triples"] = [t for t in r["hypothesis_triples"] if t[1] in ie]
        rows.append(r)
    return rows


def main(node_features="minilm", epochs=15, hidden=128, lr=1e-3, batch_size=64,
         train_path=TRAIN_PATH, test_path=TEST_PATH, verbose=True):
    """Entraîne le RGAT (train = SNLI validation, test = SNLI test) et renvoie un dict.

    node_features : 'minilm' (sémantique des mots) ou 'random' (ablation : structure seule).
    Retour : {history: [(epoch, loss, acc, f1)...], y_true, y_pred, baseline, ...}.
    """
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"device: {device} | node_features={node_features}")

    with open(os.path.join(_HERE, "data", "relations.json")) as f:
        ie = json.load(f)["ie"]
    rel2id = {r: i for i, r in enumerate(ie)}
    rel2id[GLOBAL_REL] = len(rel2id)
    num_relations = len(rel2id)

    train_rows = load_rows(train_path, set(ie))
    test_rows = load_rows(test_path, set(ie))
    if verbose:
        print(f"train {len(train_rows)} (valid) / test {len(test_rows)}")

    encoder = FeatureEncoder(mode=node_features, device=device)
    train_ds = NLIGraphDataset(train_rows, rel2id, encoder)
    test_ds = NLIGraphDataset(test_rows, rel2id, encoder)
    follow = ["p_x", "h_x"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, follow_batch=follow)
    test_loader = DataLoader(test_ds, batch_size=128, follow_batch=follow)

    model = RGAT(num_relations, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    def evaluate(loader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                ys += data.y.view(-1).tolist(); ps += model(data).argmax(1).tolist()
        return ys, ps

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for data in train_loader:
            data = data.to(device)
            opt.zero_grad()
            loss = crit(model(data), data.y.view(-1))
            loss.backward(); opt.step()
            total += loss.item() * data.num_graphs
        ys, ps = evaluate(test_loader)
        acc, f1 = accuracy_score(ys, ps), f1_score(ys, ps, average="macro")
        history.append((epoch, total / len(train_ds), acc, f1))
        if verbose:
            print(f"epoch {epoch:2d} | loss {total/len(train_ds):.4f} | "
                  f"test acc {acc:.4f} | macro-F1 {f1:.4f}")

    ys, ps = evaluate(test_loader)
    names = ["entailment", "neutral", "contradiction"]
    baseline = max(np.bincount(ys)) / len(ys)
    if verbose:
        print("\n=== rapport final (test) ===")
        print(classification_report(ys, ps, target_names=names, digits=3))
        print("matrice de confusion (lignes=vrai, cols=préd):")
        print(confusion_matrix(ys, ps))
        print(f"\nbaseline classe majoritaire: {baseline:.3f}")

    return {
        "model": model, "history": history, "y_true": ys, "y_pred": ps,
        "labels": names, "baseline": baseline,
        "best_acc": max(h[2] for h in history),
        "confusion": confusion_matrix(ys, ps),
    }


if __name__ == "__main__":
    main()
