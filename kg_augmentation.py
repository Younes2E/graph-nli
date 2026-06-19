from sentence_transformers import SentenceTransformer, util, CrossEncoder
import pandas as pd
import numpy as np
import re

# TODO : Faire une seul classe (choisir le type d'encoder), pour fusionner la recherche, et implementer le calcul de similarité a chaque saut. + Garder les méthodes naïves

class ConceptGraph():
    def __init__(self, lang):
        self.data = pd.read_csv(f'data/concept_net_{lang}.csv')

    def get_rel_forward(self, head, rel=None, sorted=True):
        if isinstance(head, str):
            head = [head]
            
        mots_echappes = [re.escape(mot) for mot in head]
        pattern = rf"^(?:{'|'.join(mots_echappes)})(?:/|$)"
        
        mask = self.data['head'].str.match(pattern, na=False)
        
        if rel is not None:
            if isinstance(rel, str):
                rel = [rel]
            mask = mask & self.data['relation'].isin(rel)
            
        result = self.data.loc[mask]
        
        return result.sort_values(by="weight", ascending=False) if sorted else result

    def get_rel_backward(self, tail, rel=None, sorted=True):
        if isinstance(tail, str):
            tail = [tail]
            
        mots_echappes = [re.escape(mot) for mot in tail]
        pattern = rf"^(?:{'|'.join(mots_echappes)})(?:/|$)"
        
        mask = self.data['tail'].str.match(pattern, na=False)
        
        if rel is not None:
            if isinstance(rel, str):
                rel = [rel]
            mask = mask & self.data['relation'].isin(rel)
            
        result = self.data.loc[mask]
        
        return result.sort_values(by="weight", ascending=False) if sorted else result
        
    def get_rel_from_triples(self, triples, dist=1, forward=True):
        current_entities = list(set([t[0] for t in triples] + [t[2] for t in triples]))
        
        collected_dfs = []
        
        for d in range(dist):
            if len(current_entities) == 0:
                break
                
            if forward:
                hop_df = self.get_rel_forward(current_entities, sorted=False)
                current_entities = hop_df['tail'].unique()
            else:
                hop_df = self.get_rel_backward(current_entities, sorted=False)
                current_entities = hop_df['head'].unique()
                
            collected_dfs.append(hop_df)
            
        if not collected_dfs:
            return np.empty((0, 4))
            
        final_df = pd.concat(collected_dfs, ignore_index=True).drop_duplicates()
        
        return final_df.sort_values(by='weight', ascending=False)[['head', 'relation', 'tail', 'weight']].to_numpy()
    

class CrossEncoderSimilarity():
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')


    def compare_sentence_triples(self, sentence, triples):
        triples_text = [f"{line[0]} {line[1]} {line[2]}" for line in triples]

        pairs = [[sentence, text] for text in triples_text]
        scores = self.model.predict(pairs)

        data = np.column_stack((triples, scores))
        index_sorted = np.argsort(data[:, 4].astype(float))[::-1]

        return data[index_sorted]
    
    
class BiEncoderSimilarity():
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


    def compare_sentence_triples(self, sentence, triples):
        triples_text = [f"{line[0]} {line[1]} {line[2]}" for line in triples]

        embedding_sentence = self.model.encode(sentence, convert_to_tensor=True)
        embeddings_triples = self.model.encode(triples_text, convert_to_tensor=True)

        similarites = util.cos_sim(embedding_sentence, embeddings_triples)
        scores = similarites[0].cpu().numpy()

        data = np.column_stack((triples, scores))
        index_sorted = np.argsort(data[:, 4].astype(float))[::-1]

        return data[index_sorted]
    

