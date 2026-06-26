from sentence_transformers import SentenceTransformer, util, CrossEncoder
from utils.graph import KnowledgeGraph
import pandas as pd
import numpy as np
import json
import re

# TODO : Faire une seul classe (choisir le type d'encoder), pour fusionner la recherche, et implementer le calcul de similarité a chaque saut. + Garder les méthodes naïves

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

class ConceptGraph():
    def __init__(self, lang, device = "cpu"):
        #self.model = SentenceTransformer(MODEL_NAME, device=device)
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device = device)
        self.data = pd.read_csv(f'data/concept_net_{lang}.csv')
        with open('data/relations.json', encoding='utf-8') as file:
            relations_raw = json.load(file)
            self.sentence_to_relation  = {item["sentence"] : item["relation"] for item in relations_raw["definitions"]}
            self.relation_to_sentence  = {item["relation"] : item["sentence"] for item in relations_raw["definitions"]}


    def get_rel_nodes(self, nodes, rel=None, sorted=True, forward = True):
        mots_echappes = [re.escape(mot) for mot in nodes]
        pattern = rf"^(?:{'|'.join(mots_echappes)})(?:/|$)"
        
        mask = self.data['head' if forward else 'tail'].str.match(pattern, na=False)
        
        if rel is not None:
            mask = mask & self.data['relation'].isin(rel)
            
        result = self.data.loc[mask]
        
        return result.sort_values(by="weight", ascending=False) if sorted else result
    
    def get_nodes_rel(self, rel, nodes = None, sorted = True, forward = True):
        mask = self.data['relation'].isin(rel)
        if nodes is not None:
            mots_echappes = [re.escape(mot) for mot in nodes]
            pattern = rf"^(?:{'|'.join(mots_echappes)})(?:/|$)"
            mask = mask & self.data['head' if forward else 'tail'].str.match(pattern, na=False)

        result = self.data.loc[mask]
        return result.sort_values(by="weight", ascending=False) if sorted else result

    def get_rel_naive(self, graph : KnowledgeGraph, dist=1, rel : list = None, entities = None, forward=True):
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
    
    def get_rel_difference(self, source : KnowledgeGraph, target : KnowledgeGraph, dist=1, rel : list = None, forward = True):
        source_entities = source.get_entities()
        target_entities = target.get_entities() - source_entities
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

    def _score_path(scores, alpha = 0.5):
        return alpha*np.mean(scores)+(1-alpha)*np.max(scores)
    

    """
    def _reconstruct_path(dfs):

    def _snap_entity(self, entity, ):
    """


    
    def _triples_to_sentences(self, triples):
        sentences = []
        for t in triples:
            sentences.append(f"{t[0]} {self.relation_to_sentence[t[1]]} {t[2]}")
        return sentences
    
    def similarity_score(self, link, sentences):
        x = [(link, s) for s in sentences]
        scores = self.model.predict(x)
        data_score = np.column_stack((sentences, scores))
        index_sorted = np.argsort(data_score[:, 1].astype(float))[::-1]
        return data_score[index_sorted]

    def get_rel_similarity(self, source : KnowledgeGraph, target : KnowledgeGraph, dist=1, rel : list = None, forward = True, threshold = None):
        ## TODO : Faire le score par rapport a H - P, pour encoder la difference !!!!!!!!!!!!!!!!!!
        """ TODO : description bridge dans relations.json (eg. "is a form of"), puis concat e1+phrase+e2 puis calculer le score
        Au lieu de comparer avec la phrase complete on va comparer avec les triplets (mis sous forme de phrase) non present dans source. Pour ne pas comparer des informations identiques
        Pour la fonction get_rel finale : A chaque iter (range(d)) forward puis snap
        A la fin : reconstruct path, puis score chaque path (avec cette fonction) et renvoyer path et score

        Piste à tester : comparer a chacun des triplets (sous forme de phrase) de H et prendre le highest score (faire ça à toutes les itérations jusqu'a ce qu'un reel ecart se creuse, et dans ce cas là pour chaque extension, on va la comparer a ce meme triplet argmax)
        Il faut partager le travail et pas que tous les chemins focus la meme  information de target. Il faut peut etre partir de chaque info de target et faire un forward=False, renvoyer le score avec l'info qu'il cherche (l'argmax du score)
        """

        source_entities = source.get_entities()
        ## Entités de depart (peut etre prendre les entité presentes dans (P.rel - H.rel))

        target_sentences = self._triples_to_sentences(target.get_rel() - source.get_rel())
        ## Triplets non resolues de target sous forme de phrase pour le cross encodeur

        ## for _ in range d : Forward
        ###### Entity snap
        ###### Score link forwardé avec target_sentences

        ## Reconstruct les paths
        ## Scorer les paths

        # renvoyer les paths avec un score superieur a threshold (2 df, un df avec [id_path, e1, rel, e2] l'autre avec [id_path, score_path])
