from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
import spacy
import json
import re
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")

PROP_MODEL_NAME = "Zual/MPropositioneur-V2-large"
TRIPLES_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


class GraphBuilder:
    def __init__(self, gpu_mem_total=None, max_model_len=4096,
                 enforce_eager=True, kv_cache_gb=2.0, **llm_kwargs):
        self._gpu_mem_total = gpu_mem_total
        self._kv_cache_gb = kv_cache_gb
        self._llm_common = dict(dtype="float16", max_model_len=max_model_len,
                                enforce_eager=enforce_eager, **llm_kwargs)
        self._loaded = None
        self.tokenizer = None
        self.llm = None

        self.nlp = spacy.load("en_core_web_sm")

        with open(os.path.join(_DATA_DIR, "relations.json"), encoding="utf-8") as file:
            relations_raw = json.load(file)
            self.ie = set(relations_raw["ie"])
            self.relations = [
                {k: v for k, v in item.items() if k in {"relation", "description"}}
                for item in relations_raw["definitions"] if item["relation"] in self.ie
            ]

        with open(os.path.join(_DATA_DIR, "exemples.json"), encoding="utf-8") as file:
            exemples_raw = json.load(file)
            lines = []
            for line in exemples_raw:
                lines.append(f'Input: "{line["input"]}"')
                lines.append("Output:")
                for triple in line["output"]:
                    lines.append(" | ".join(triple))
                lines.append("\n")
            self.exemples = "\n".join(lines)

    def _util_for(self, weights_gb):
        if self._gpu_mem_total is not None:
            return self._gpu_mem_total
        import torch
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return min(0.95, (weights_gb + self._kv_cache_gb) / total_gb)

    def load(self, model):
        """Charge "prop" ou "qwen" en VRAM (décharge l'autre si besoin)."""
        if self._loaded == model:
            return
        self.unload()
        from vllm import LLM
        name, weights = (PROP_MODEL_NAME, 8.5) if model == "prop" else (TRIPLES_MODEL_NAME, 15.0)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.llm = LLM(model=name, gpu_memory_utilization=self._util_for(weights),
                       **self._llm_common)
        self._loaded = model

    def unload(self):
        """Libère la VRAM du modèle chargé."""
        if self.llm is None:
            return
        del self.llm
        self.llm = self.tokenizer = None
        self._loaded = None
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    def extract_atomic_prop_batch(self, texts):
        from vllm import SamplingParams
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"Atomize: {t}"}],
                tokenize=False, add_generation_prompt=True)
            for t in texts
        ]
        outputs = self.llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=2048))
        return [set(self._parse_props(o.outputs[0].text)) for o in outputs]

    @staticmethod
    def _parse_props(text):
        text = text.strip()
        try:
            return list(dict.fromkeys(json.loads(text)))
        except Exception:
            pass
        out, seen = [], set()
        for s in re.findall(r'"((?:[^"\\]|\\.)*)"', text):
            if "\\" in s:
                try:
                    s = json.loads(f'"{s}"')
                except Exception:
                    pass
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def extract_triples_batch(self, props):
        from vllm import SamplingParams
        prompts = [self._triples_prompt(p) for p in props]
        outputs = self.llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=512))
        return [self._parse_triples(o.outputs[0].text) for o in outputs]

    def _triples_prompt(self, text):
        prompt = f"""
        Extract all factual ([SUBJECT], [RELATION], [OBJECT]) triples from sentence.
        One triple per line in the format:
        [SUBJECT] | [RELATION] | [OBJECT]

        No explanations. If no triple can be extracted, write nothing.

        ALLOWED RELATIONS:
        {json.dumps(self.relations, indent=1)}
        Do NOT use any relation outside of this list.

        ADDITIONAL RULES:
        1. If an action is negated in the sentence (e.g., "didn't call", "is not eating"), you MUST capture the negation inside the [VERB] node using 'not' (e.g., 'not call', 'not eating').
        2. When an action involves multiple elements at once (e.g., an actor, a target, a recipient, a tool, or a location), do NOT link the secondary elements to each other. Instead, make the action the central hub and ALL links must involve the action.
        3. NODES MUST BE ATOMIC. Each node is a single, indivisible concept (normally one word). 
            Never keep a concept together with its modifiers as one node: 
            split off every modifier/descriptor, owner, quantity, material, qualifying noun, etc as its own triple, with the core concept as the node and the most appropriate relation from the list above. 
            Examples:
                "old man"      -> [man, HasProperty, old]
                "wooden table" -> [table, MadeOf, wood]
                "two kidneys"  -> [kidneys, HasQuantity, two]
                "ham sandwich" -> [sandwich, Contains, ham]
            EXCEPTION: proper nouns and fixed/lexicalized compounds stay as ONE node — never
            split them: "New York", "ice cream", "United States", "Eiffel Tower", "hot dog", "Micheal Jackson".
            
        EXTRACTION EXEMPLES:
        {self.exemples}


        Sentence: {text}"""

        messages = [
            {"role": "system", "content": "You are a deterministic Information Extraction expert."},
            {"role": "user", "content": prompt},
        ]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def _lemmatize_entity(self, text):
        doc = self.nlp(text)
        keep = [
            t for t in doc
            if t.pos_ not in {"DET", "ADP", "CCONJ", "SCONJ", "PUNCT", "PART", "AUX"}
            or t.lemma_.lower() in {"not", "no", "never"}
        ]
        if not keep:
            keep = list(doc)
        lemmas = [t.lemma_.lower() for t in keep]
        return "_".join(lemmas)

    def _parse_triples(self, content):
        triples = []
        for line in content.split("\n"):
            line = line.strip().strip("`").strip()
            if "|" not in line:
                continue
            parts = [c.strip().strip("[]").strip() for c in line.split("|")]
            if len(parts) != 3 or not all(parts):
                continue
            subj, rel, obj = parts
            if rel not in self.ie:
                continue
            triples.append((self._lemmatize_entity(subj),
                            rel,
                            self._lemmatize_entity(obj)))
        return triples

    def build_batch(self, texts, batch_size=512, show_progress=True):
        texts = list(texts)

        def chunks(seq):
            for i in range(0, len(seq), batch_size):
                yield seq[i:i + batch_size]

        self.load("prop")
        props_per_text = []
        it = chunks(texts)
        if show_progress:
            it = tqdm(it, total=(len(texts) + batch_size - 1) // batch_size, desc="prop")
        for batch in it:
            props_per_text.extend(self.extract_atomic_prop_batch(batch))

        flat_props, owners = [], []
        for i, props in enumerate(props_per_text):
            for p in props:
                flat_props.append(p)
                owners.append(i)

        self.load("qwen")
        triples_per_prop = []
        it = chunks(flat_props)
        if show_progress:
            it = tqdm(it, total=(len(flat_props) + batch_size - 1) // batch_size, desc="triples")
        for batch in it:
            triples_per_prop.extend(self.extract_triples_batch(batch))
        self.unload()

        per_text = [set() for _ in texts]
        for owner, triples in zip(owners, triples_per_prop):
            per_text[owner].update(triples)
        return [KnowledgeGraph(t, tr) for t, tr in zip(texts, per_text)]

    def build(self, text):
        return self.build_batch([text], show_progress=False)[0]


class KnowledgeGraph:
    def __init__(self, text, triples):
        self.text = text
        self.triples_sentence = set(triples)
        self.triples_augmented = set()
        self.entities = {t[i]: True for t in triples for i in (0, 2)}

    def augment(self, data):
        triples = [(h, r, t) for h, r, t in data[['head', 'rel', 'tail']].itertuples(index=False, name=None)]
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

    def get_rel(self, sentence = None):
        if sentence is None:
            return self.triples_sentence | self.triples_augmented
        elif isinstance(sentence, bool):
            return self.triples_sentence if sentence else self.triples_augmented
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
