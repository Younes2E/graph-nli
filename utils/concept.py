from sentence_transformers import SentenceTransformer, util, CrossEncoder
from utils.graph import KnowledgeGraph
import pandas as pd
import numpy as np
import json
import os


CROSS_NAME = 'cross-encoder/stsb-roberta-base'

class ConceptGraph():
    def __init__(self, lang = 'en', device = "cuda"):
        self.device = device
        self.cross = CrossEncoder(CROSS_NAME, device=self.device)
        
        data_raw = pd.read_csv(f'data/concept_net_{lang}.csv')
        self.data = data_raw[['head_word', 'relation', 'tail_word', 'weight']]

        self.fwd = {w: g for w, g in self.data.groupby('head_word', sort=False)}
        self.bwd = {w: g for w, g in self.data.groupby('tail_word', sort=False)}
        self.empty = self.data.iloc[0:0]

        with open('data/relations.json', encoding='utf-8') as file:
            relations_raw = json.load(file)
            self.sentence_to_relation  = {item["sentence"] : item["relation"] for item in relations_raw["definitions"]}
            self.relation_to_sentence  = {item["relation"] : item["sentence"] for item in relations_raw["definitions"]}

    def get_rel_nodes(self, nodes, rel=None, sorted=True, forward = True):
        index = self.fwd if forward else self.bwd
        parts = [index[n] for n in set(nodes) if n in index]
        result = pd.concat(parts) if parts else self.empty

        if rel is not None:
            result = result[result['relation'].isin(rel)]

        return result.sort_values(by="weight", ascending=False) if sorted else result

    def get_nodes_rel(self, rel, nodes = None, sorted = True, forward = True):
        if nodes is None:
            result = self.data[self.data['relation'].isin(rel)]
        else:
            index = self.fwd if forward else self.bwd
            parts = [index[n] for n in set(nodes) if n in index]
            result = pd.concat(parts) if parts else self.empty
            result = result[result['relation'].isin(rel)]

        return result.sort_values(by="weight", ascending=False) if sorted else result

    def get_rel_naive(self, graph : KnowledgeGraph, dist=1, rel : list = None, entities = None, forward=True):
        current_entities = graph.get_entities()
        dfs = []
        next_col = 'tail_word' if forward else 'head_word'

        for _ in range(dist):
            if len(current_entities) == 0:
                break

            hop = self.get_rel_nodes(current_entities, rel = rel,sorted=False, forward = forward)
            current_entities = hop[next_col].unique()

            dfs.append(hop)

        if not dfs:
            return np.empty((0, 4))

        final_df = pd.concat(dfs, ignore_index=True).drop_duplicates()

        return final_df.sort_values(by='weight', ascending=False)[['head_word', 'relation', 'tail_word', 'weight']].to_numpy()

    def get_rel_difference(self, source : KnowledgeGraph, target : KnowledgeGraph, dist=1, rel : list = None, forward = True):
        source_entities = source.get_entities()
        target_entities = target.get_entities() - source_entities
        next_col = 'tail_word' if forward else 'head_word'
        prev_col = 'head_word' if forward else 'tail_word'
        dfs = []

        for _ in range(dist):
            if len(source_entities) == 0 or len(target_entities) == 0:
                break
            hop = self.get_rel_nodes(source_entities, rel = rel,sorted=False, forward = forward)
            if hop.empty:
                break

            source_entities = set(hop[next_col].unique())
            dfs.append(hop)

        if not dfs:
            return np.empty((0, 4))

        active_targets = target_entities
        final_dfs = []

        for current_df in reversed(dfs):
            if len(active_targets) == 0:
                break

            mask = current_df[next_col].isin(active_targets)
            hop = current_df[mask]

            if not hop.empty:
                final_dfs.append(hop)
                active_targets.update(set(hop[prev_col].unique()))

        if not final_dfs:
            return np.empty((0, 4))

        final_df = pd.concat(final_dfs, ignore_index=True).drop_duplicates()
        return final_df.sort_values(by='weight', ascending=False)[['head_word', 'relation', 'tail_word', 'weight']].to_numpy()

    @staticmethod
    def _score_path(scores, alpha = 0.5):
        return alpha*np.mean(scores)+(1-alpha)*np.max(scores)
    
    def _reconstruct_paths(self, hop, threshold= 0.5):
        edge_cols = ['id_path', 'head', 'rel', 'tail', 'score_link', 'info_scored']
        path_cols = ['id_path', 'score_path']
        if not hop or hop[0].empty:
            return pd.DataFrame(columns=edge_cols), pd.DataFrame(columns=path_cols)

        def edges_of(df): 
            return list(df[['head_word', 'relation', 'tail_word', 'score', 'info_scored']]
                        .itertuples(index=False, name=None))

        frontier = [([e], e[2], {e[0], e[2]}) for e in edges_of(hop[0])]
        completed = []

        for k in range(1, len(hop)):
            by_head = {}
            for e in edges_of(hop[k]):
                by_head.setdefault(e[0], []).append(e)

            new_frontier = []
            for path, node, visited in frontier:
                if path[-1][3] == 1.0:                 
                    completed.append(path); continue
                exts = [e for e in by_head.get(node, []) if e[2] not in visited]  
                if not exts:
                    completed.append(path)             
                else:
                    for e in exts:
                        new_frontier.append((path + [e], e[2], visited | {e[2]}))
            frontier = new_frontier
        completed.extend(p for p, _, _ in frontier) 

        scored = [(self._score_path([e[3] for e in path]), path) for path in completed]
        scored = [sp for sp in scored if sp[0] >= threshold]
        scored.sort(key=lambda x: x[0], reverse=True)

        edge_rows, path_rows = [], []
        for pid, (ps, path) in enumerate(scored):
            path_rows.append((pid, ps))
            for (h, r, t, sc, info) in path:
                edge_rows.append((pid, h, r, t, sc, info))

        return (pd.DataFrame(edge_rows, columns=edge_cols),
                pd.DataFrame(path_rows, columns=path_cols))

    def _triples_to_sentences(self, triples):
        return [f"{h} {self.relation_to_sentence.get(r, r)} {t}" for (h, r, t, *_) in triples]

    def _max_scores(self, links, sentences):
        links = list(links); sentences = list(sentences)
        if not links or not sentences:
            return np.zeros(len(links)), [None] * len(links)
        pairs = [(l, s) for l in links for s in sentences]
        sc = np.asarray(self.cross.predict(pairs), dtype=float).reshape(len(links), len(sentences))
        j = sc.argmax(axis=1)
        return sc[np.arange(len(links)), j], [sentences[k] for k in j]

    def get_relevant_data(self, source, target, threshold_link = 0.3, threshold_path= 0.5, forward = True, dist = 1):
        # Garder le seuil pour les chemins, mais faire un top_k (proportionnel à unresolved) pour garder les liens ?
        ## TODO ?? : Avoir un poid different en fonction de la rel ? par exemple 'IsA' est presque obligatoire et systematique, 
        # IsA permet de faire un premier pont, ensuite on prune les chemins 'IsA' qui n'ont rien de pertinent (score) qui partent d'eux

        cols = ['head_word', 'relation', 'tail_word', 'score', 'info_scored']

        unresolved = self._triples_to_sentences(target.get_rel(sentence=True) - source.get_rel(sentence=True)) ## Soustraire 'e' & 'rel' si P(x, rel, e) et H(y, rel, e) ??
        if not unresolved:
            return self._reconstruct_paths([])

        start_entities = source.get_entities(sentence=True) - target.get_entities(sentence=True) #set([t[i] for t in (H.get_rel() - P.get_rel()) for i in (0,2)])
        target_entities = target.get_entities(sentence=True) - source.get_entities(sentence=True)

        next_col = 'tail_word' if forward else 'head_word'
        prev_col = 'head_word' if forward else 'tail_word'

        hops = []
        for _ in range(dist):
            if not start_entities :
                break

            df = self.get_rel_nodes(start_entities, forward= forward).copy()
            if df.empty :
                break

            match_mask = df[next_col].isin(target_entities)
            df['score'] = 0.0
            df['info_scored'] = None

            df.loc[match_mask, 'score'] = 1.
            df.loc[match_mask, 'info_scored'] = df.loc[match_mask, next_col]


            rest = df.loc[~match_mask]
            if len(rest):
                sents = self._triples_to_sentences([(h, r, t) for h, r, t in rest[['head_word', 'relation', 'tail_word']].itertuples(index=False, name=None)])
                scores, best = self._max_scores(sents, unresolved)
                df.loc[~match_mask, 'score'] = scores
                df.loc[~match_mask, 'info_scored'] = np.array(best, dtype=object)

            df = df[df['score'] >= threshold_link]
            if df.empty:
                break
            hops.append(df)
            
            start_entities = set(df[next_col]) - target_entities
        
        return self._reconstruct_paths(hops, threshold= threshold_path)