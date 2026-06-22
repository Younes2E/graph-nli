from sentence_transformers import SentenceTransformer, util
from graph import KnowledgeGraph
import pandas as pd
import numpy as np
import re

# TODO : Faire une seul classe (choisir le type d'encoder), pour fusionner la recherche, et implementer le calcul de similarité a chaque saut. + Garder les méthodes naïves

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

class ConceptGraph():
    def __init__(self, lang, device = "cpu"):
        self.model = SentenceTransformer(MODEL_NAME, device=device)
        self.data = pd.read_csv(f'data/concept_net_{lang}.csv')

    def get_rel_nodes(self, nodes, rel=None, sorted=True, forward = True):
        mots_echappes = [re.escape(mot) for mot in nodes]
        pattern = rf"^(?:{'|'.join(mots_echappes)})(?:/|$)"
        
        mask = self.data['head' if forward else 'tail'].str.match(pattern, na=False)
        
        if rel is not None:
            mask = mask & self.data['relation'].isin(rel)
            
        result = self.data.loc[mask]
        
        return result.sort_values(by="weight", ascending=False) if sorted else result
 
    def get_rel_naive(self, graph : KnowledgeGraph, dist=1, rel = None, entities = None, forward=True):
        current_entities = graph.get_entities()
        dfs = []
        next_col = 'tail' if forward else 'head'
        
        for _ in range(dist):
            if len(current_entities) == 0:
                break

            hop = self.get_rel_nodes(current_entities, rel = rel,sorted=False, forward = forward)
            current_entities = hop[next_col].str.split('/').str[0].unique()

            dfs.append(hop)

        if not dfs:
            return np.empty((0, 4))

        final_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
        
        return final_df.sort_values(by='weight', ascending=False)[['head', 'relation', 'tail', 'weight']].to_numpy()
    
    
    def similarity_score(self, data, target):
        if len(data) == 0:
            return np.empty((0, 5)) 
        
        triples_text = [f"{line[0]} {line[1]} {line[2]}" for line in data]

        embedding_sentence = self.model.encode(target.get_text(), convert_to_tensor=True)
        embeddings_triples = self.model.encode(triples_text, convert_to_tensor=True)

        similarites = util.cos_sim(embedding_sentence, embeddings_triples)
        scores = similarites[0].cpu().numpy()

        data_score = np.column_stack((data, scores))
        index_sorted = np.argsort(data_score[:, 4].astype(float))[::-1]

        return data_score[index_sorted]
    


    def get_rel_difference(self, source, target, dist=1, rel = None, forward = True):
        source_entities = source.get_entities()
        target_entities = target.get_entities() - source_entities ## TODO : changer ça
        next_col = 'tail' if forward else 'head'
        prev_col = 'head' if forward else 'tail'
        dfs = []

        for _ in range(dist):
            if len(source_entities) == 0 or len(target_entities) == 0:
                break
            hop = self.get_rel_nodes(source_entities, rel = rel,sorted=False, forward = forward).copy()
            if hop.empty:
                break

            hop['head'] = hop['head'].str.split('/').str[0]
            hop['tail'] = hop['tail'].str.split('/').str[0]

            source_entities = set(hop[next_col].str.split('/').str[0].unique())
            dfs.append(hop) 

        if not dfs:
            return np.empty((0, 4))
        
        active_targets = target_entities
        final_dfs = []

        for current_df in reversed(dfs):
            if len(active_targets) == 0:
                break
            
            mask = current_df[next_col].isin(active_targets)
            hop = current_df[mask].copy()

            if not hop.empty:
                final_dfs.append(hop)
                active_targets.update(set(hop[prev_col].unique()))

        if not final_dfs:
            return np.empty((0, 4))

        final_df = pd.concat(final_dfs, ignore_index=True).drop_duplicates()  
        return final_df.sort_values(by='weight', ascending=False)[['head', 'relation', 'tail', 'weight']].to_numpy()


        

