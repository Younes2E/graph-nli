from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import networkx as nx
import torch
import json
import os

class GraphBuilder:
    def __init__(self, device = "cpu"):
        self.device = device

        self.prop_name = "Zual/MPropositioneur-V2-large"
        self.prop_tokeniser = AutoTokenizer.from_pretrained(self.prop_name)
        self.prop_model = AutoModelForCausalLM.from_pretrained(self.prop_name, dtype=torch.float16).to(self.device)

        self.qwen_name = "Qwen/Qwen2.5-7B-Instruct"
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(self.qwen_name)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(self.qwen_name, dtype=torch.float16).to(self.device)

        with open('data/relations.json', encoding='utf-8') as file:
            self.relations = json.load(file)

        with open('data/exemples.json', encoding='utf-8') as file:
            exemples_raw = json.load(file)
            lines = []
            for line in exemples_raw:
                lines.append(f'Input: "{line["input"]}"')
                lines.append("Output:")
                for triple in line['output']:
                    lines.append(" | ".join(triple))
                lines.append("\n")
            self.exemples = "\n".join(lines)

    def extract_atomic_prop(self, text):  
        prompt = f"<|im_start|>user\nAtomize: {text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = self.prop_tokeniser(prompt, return_tensors="pt", truncation=True, max_length=8192).to(self.device)
        
        with torch.no_grad():
            outputs = self.prop_model.generate(**inputs, max_new_tokens=2048, do_sample=False)

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        result = self.prop_tokeniser.decode(generated_ids, skip_special_tokens=True).strip()
        
        propositions = json.loads(result)
        
        return set(propositions)

    def extract_triples(self, text):
        prompt = f"""
        Extract all factual ([SUBJECT], [RELATION], [OBJECT]) triples from sentence. 
        One triple per line in the format: 
        [SUBJECT] | [RELATION] | [OBJECT]

        No explanations. If no triple can be extracted, write nothing. 

        ALLOWED RELATIONS:
        {json.dumps(self.relations, indent=1)}
        Do NOT use any relation outside of this list.

        ADDITIONAL RULES: 
        1. If an action is negated in the sentence (e.g., "didn't call", "is not eating"), you MUST capture the negation inside the [VERB] node using 'not' (e.g., 'not_call', 'not_eating').
        2. When an action involves multiple elements at once (e.g., an actor, a target, a recipient, a tool, or a location), do NOT link the secondary elements to each other. Instead, make the action the central hub and ALL links must involve the action.

        EXEMPLES:
        {self.exemples}

        
        Sentence: {text}"""

        messages = [
            {"role": "system", "content": "You are a deterministic Information Extraction expert."},
            {"role": "user", "content": prompt}
        ]
        text = self.qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            do_sample=False
            ##enable_thinking=False

        )
        model_inputs = self.qwen_tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.qwen_model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        content = self.qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        triples = []
        for line in content.split("\n"):
            t = line.split(" | ")
            try :
                triples.append((t[0].strip().lower().replace(' ', '_'), t[1].strip(), t[2].strip().lower().replace(' ', '_')))
            except : 
                print(f"\nMauvais format de triplet : {t}.\n")

        return triples

    def build(self, text): 
        propositions = self.extract_atomic_prop(text)
        triplets_qwen = []
        for p in propositions:
            triplets_qwen += self.extract_triples(p)
                
        return KnowledgeGraph(text, set(triplets_qwen))
    

class KnowledgeGraph:
    def __init__(self, text, triples):
        self.text = text
        self.triples_sentence = triples
        self.triples_augmented = set()
        self.entities = {t[i]: True for t in triples for i in (0, 2)}

    def augment(self, triples):
        self.triples_augmented.update(triples)        
        for t in triples:
            self.entities.setdefault(t[0], False)
            self.entities.setdefault(t[2], False)

    def get_entities(self, sentence = None):
        # sentence : si ce sont des nodes qui viennent du text original ou augmented
        if sentence is None:
            return set(self.entities.keys())
        elif isinstance(sentence, bool):
            return set([ent for ent, v in self.entities.items() if v is sentence])
        else:
            raise ValueError("'sentence' must be True, False or None.")
    
    def get_text(self):
        return self.text

    def display(self, file=None):
        G = nx.MultiDiGraph()

        for subj, rel, obj in self.triples_sentence:
            G.add_edge(subj, obj, relation=rel, origin='base')

        for subj, rel, obj in self.triples_augmented:
            exists_in_base = False
            if G.has_edge(subj, obj):
                for key, data in G[subj][obj].items():
                    if data['relation'] == rel and data['origin'] == 'base':
                        G[subj][obj][key]['origin'] = 'both'
                        exists_in_base = True
                        break

            if not exists_in_base:
                G.add_edge(subj, obj, relation=rel, origin='augmented')

        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(G, k=2.0, seed=42)

        node_colors = []
        for node in G.nodes():
            if self.entities.get(node) is True:
                node_colors.append("skyblue")
            else:
                node_colors.append("lightcoral")

        nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_family="sans-serif")

        pair_counts = {}
        for u, v, k, d in G.edges(keys=True, data=True):
            pair = tuple(sorted([u, v]))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        pair_seen = {}

        RAD_STEP = 0.4

        for u, v, k, d in G.edges(keys=True, data=True):
            pair = tuple(sorted([u, v]))
            idx = pair_seen.get(pair, 0)
            pair_seen[pair] = idx + 1
            total = pair_counts[pair]

            if total == 1:
                rad = 0.0  
            else:
                rad = (idx - (total - 1) / 2) * RAD_STEP
                if u > v:  
                    rad = -rad

            if d['origin'] == 'base':
                color = 'black'
                style = 'solid'
            elif d['origin'] == 'augmented':
                color = 'red'
                style = 'solid'
            else:
                color = 'purple'
                style = 'dashed'

            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                arrowstyle="->", arrowsize=20,
                edge_color=color, width=1.5, style=style,
                min_source_margin=30, min_target_margin=30,
                connectionstyle=f"arc3,rad={rad}"
            )

            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels={(u, v): d['relation']},
                font_size=8, font_color=color,
                label_pos=0.5,
                connectionstyle=f"arc3,rad={rad}"
            )

        title_text = self.text if hasattr(self, 'text') else "Knowledge Graph"
        plt.title(title_text, fontsize=10, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()

        if file:
            plt.savefig(file, format="png", dpi=300)
            plt.close()
            print(f"Graphe sauvegardé avec succès dans : {os.path.abspath(file)}")
        else:
            plt.show()
            plt.close()